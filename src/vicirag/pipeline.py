"""Ghép ICI + ACMG thành pipeline ViCiRAG hoàn chỉnh và chạy đánh giá.

Cách dùng:
    python -m vicirag.pipeline --dataset vimqa_sample --split dev --it 2
    python -m vicirag.pipeline --dataset vimqa --split dev --ab fact       # ablation Table 3
    python -m vicirag.pipeline --dataset vimqa --split dev --no-reranker   # ablation Table 2
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .acmg import AdaptiveCascadedGeneration
from .baselines.ircot import IRCoT
from .baselines.native_rag import NativeRAG
from .config import Config
from .data import Sample, load_corpus, load_samples
from .embedding.vi_encoder import VietnameseEncoder
from .eval.vi_metrics import evaluate, retrieval_recall
from .ici import IterativeConstructionIntegration
from .llm.openrouter import OpenRouterLLM, TokenUsage
from .openie import OpenIEExtractor, normalize_triples_official
from .retriever import Retriever

#  Ánh xạ cờ --ab của repo gốc sang chuỗi mức ngữ cảnh của ACMG.
AB_TO_CASCADE = {
    "": ("fact", "sentence", "passage"),
    "fact": ("fact",),
    "sentence": ("sentence",),
    "passage": ("passage",),
    "fact-sentence": ("fact", "sentence"),
    "fact-passage": ("fact", "passage"),
    "sentence-passage": ("sentence", "passage"),
    "fact-sentence-passage": ("fact", "sentence", "passage"),
}


class ViCiRAG:
    """Chạy CiRAG hoặc một baseline trên cùng retriever và cùng backbone LLM.

    Dùng chung một lớp cho cả ba phương pháp là có chủ đích: paper nhấn mạnh mọi
    baseline phải dùng cùng retriever, cùng corpus và cùng LLM thì so sánh mới
    công bằng (Sec 4.2).
    """

    def __init__(self, config: Config, lang: str = "vi", method: str = "cirag",
                 reason_lang: str | None = None) -> None:
        self.config = config
        self.lang = lang
        #  Ngôn ngữ prompt cho ICI và ACMG, tách khỏi ngôn ngữ trích triple. Cho phép
        #  chạy prompt tiếng Anh trên dữ liệu tiếng Việt để kiểm chứng chất lượng prompt.
        self.reason_lang = reason_lang or lang
        self.method = method
        self.usage = TokenUsage()

        self.llm = OpenRouterLLM(
            model=config.llm_name,
            base_url=config.llm_base_url,
            api_key=config.api_key,
            cache_dir=config.cache_dir,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed,
            usage=self.usage,
            context_window=config.context_window,
        )

        #  Integrator K_D dùng chung LLM với các khâu khác. Bài báo gốc thay riêng
        #  thành phần này bằng một student đã chưng cất (Trajectory Distillation);
        #  bản cài đặt này không có module đó nên mọi con số ứng với cấu hình w/o TD.
        self.integrator_llm = self.llm

        self.samples: list[Sample] = load_samples(config.samples_path)
        self.corpus = load_corpus(config.corpus_path)
        print(f"[ViCiRAG] {len(self.samples)} câu hỏi, corpus {len(self.corpus)} đoạn")

        # --- Construction Phase offline: trích triple một lần cho cả corpus ---
        #  Dùng model riêng (rẻ hơn) cho phần này; cache đặt tên theo model để đổi
        #  model không vô tình đọc nhầm triple của model cũ.
        openie_model = config.openie_llm_name or config.llm_name
        self.openie_llm = (
            self.llm
            if openie_model == config.llm_name
            else OpenRouterLLM(
                model=openie_model,
                base_url=config.llm_base_url,
                api_key=config.api_key,
                cache_dir=config.cache_dir,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                seed=config.seed,
                usage=self.usage,
                context_window=config.context_window,
            )
        )
        openie_tag = openie_model.replace("/", "_").replace(":", "_")
        openie_cache = (
            config.cache_dir / f"openie_{config.dataset}_{config.split}_{lang}_{openie_tag}.json"
        )
        extractor = OpenIEExtractor(
            self.openie_llm, lang=lang, cache_path=openie_cache, num_workers=config.num_workers
        )
        triples_by_title = extractor.run(self.corpus, force=config.force_openie_from_scratch)
        if config.triple_norm == "official":
            #  Cache OpenIE vẫn giữ triple nguyên văn; chuẩn hoá ở đây để bật/tắt được
            #  mà không phải trích lại.
            triples_by_title = normalize_triples_official(triples_by_title)

        encoder = VietnameseEncoder(
            model_name=config.embedding_model_name,
            device=config.embedding_device,
            max_length=config.embedding_max_length,
            batch_size=config.embedding_batch_size,
        )
        self.retriever = Retriever(
            corpus=self.corpus,
            triples_by_title=triples_by_title,
            encoder=encoder,
            use_hybrid=config.use_hybrid_retrieval,
            rrf_k=config.rrf_k,
            candidate_scope=config.extra.get("candidate_scope", "corpus"),
            cache_dir=config.cache_dir,
        )

        self.ici = IterativeConstructionIntegration(
            retriever=self.retriever,
            llm=self.integrator_llm,
            lang=self.reason_lang,
            top_k=config.retrieval_top_k,
            top_n=config.num_candidate_triples,
            max_iterations=config.max_iterations,
            use_reranker=config.use_triple_reranker,
            official_rescue=config.official_rescue,
        )
        self.acmg = AdaptiveCascadedGeneration(
            llm=self.llm,
            lang=self.reason_lang,
            cascade=config.granularity_cascade,
            min_source_docs=config.min_source_docs,
            passage_fallback=config.passage_fallback,
        )
        self.native_rag = NativeRAG(
            self.retriever, self.llm, lang=lang, top_k=config.retrieval_top_k
        )
        self.ircot = IRCoT(
            self.retriever,
            self.llm,
            lang=lang,
            top_k=config.retrieval_top_k,
            max_iterations=config.max_iterations,
        )

    def answer_one(self, sample: Sample) -> dict:
        if self.method == "native":
            return self._answer_baseline(sample, self.native_rag)
        if self.method == "ircot":
            return self._answer_baseline(sample, self.ircot)
        return self._answer_cirag(sample)

    def _answer_baseline(self, sample: Sample, model) -> dict:
        allowed = set(sample.own_titles)
        out = model.run(sample.question, allowed_titles=allowed)
        return {
            "id": sample.id,
            "question": sample.question,
            "gold_answer": sample.answer,
            "predicted_answer": out["predicted_answer"],
            "granularity": "passage",
            "num_steps": out["num_steps"],
            "retrieval_recall": retrieval_recall(out["retrieved_titles"], sample.gold_titles),
            "raw_response": out["raw_response"],
        }

    def _answer_cirag(self, sample: Sample) -> dict:
        allowed = set(sample.own_titles)
        ici_result = self.ici.run(sample.question, allowed_titles=allowed)
        acmg_result = self.acmg.run(sample.question, ici_result)

        retrieved = [t for log in ici_result.logs for t in log.retrieved_titles]
        return {
            "id": sample.id,
            "question": sample.question,
            "gold_answer": sample.answer,
            "predicted_answer": acmg_result.answer,
            "granularity": acmg_result.granularity,
            "num_steps": ici_result.num_steps,
            "terminated_early": ici_result.terminated_early,
            "num_core_triples": len(ici_result.core_triples),
            "num_core_sentences": len(ici_result.core_sentences),
            "num_core_passages": len(ici_result.core_passages),
            "core_triples": [list(t) for t in ici_result.core_triples],
            "retrieval_recall": retrieval_recall(retrieved, sample.gold_titles),
            "trajectory": [asdict(log) for log in ici_result.logs],
            "acmg_attempts": acmg_result.attempts,
        }

    def run(self, limit: int | None = None) -> dict:
        samples = self.samples[:limit] if limit else self.samples
        start = time.time()

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as pool:
            for result in tqdm(
                pool.map(self.answer_one, samples), total=len(samples), desc="ViCiRAG"
            ):
                results.append(result)
        elapsed = time.time() - start

        overall, per_example = evaluate(
            [r["predicted_answer"] for r in results],
            [[r["gold_answer"]] for r in results],
            lang=self.lang,
        )
        for result, scores in zip(results, per_example):
            result.update(scores)

        summary = {
            "run_name": self.config.run_name,
            "method": self.method,
            "dataset": self.config.dataset,
            "split": self.config.split,
            "llm": self.config.llm_name,
            "llm_base_url": self.config.llm_base_url,
            "integrator": getattr(self.integrator_llm, "model_name", self.config.llm_name),
            "embedding": self.config.embedding_model_name,
            "num_questions": len(results),
            "max_iterations": self.config.max_iterations,
            "cascade": list(self.config.granularity_cascade),
            "min_source_docs": self.config.min_source_docs,
            "passage_fallback": self.config.passage_fallback,
            "use_reranker": self.config.use_triple_reranker,
            "triple_norm": self.config.triple_norm,
            "official_rescue": self.config.official_rescue,
            "num_candidate_triples": self.config.num_candidate_triples,
            "use_hybrid_retrieval": self.config.use_hybrid_retrieval,
            "metrics": overall,
            "granularity_distribution": dict(Counter(r["granularity"] for r in results)),
            "avg_steps": round(sum(r["num_steps"] for r in results) / max(len(results), 1), 2),
            "avg_retrieval_recall": round(
                sum(r["retrieval_recall"] for r in results) / max(len(results), 1), 4
            ),
            "total_seconds": round(elapsed, 1),
            "seconds_per_question": round(elapsed / max(len(results), 1), 2),
            "token_usage": self.usage.to_dict(),
        }

        out_dir = self.config.output_dir / self.config.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "predictions.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        print("\n" + "=" * 62)
        print(f"Kết quả: {self.config.run_name}")
        print("=" * 62)
        print(f"  EM              {overall['em']:.2f}")
        print(f"  F1 (âm tiết)    {overall['f1']:.2f}")
        print(f"  F1 (từ)         {overall['f1_word']:.2f}")
        print(f"  Recall truy hồi {summary['avg_retrieval_recall']:.3f}")
        print(f"  Phân bố mức     {summary['granularity_distribution']}")
        print(f"  Số vòng TB      {summary['avg_steps']}")
        print(f"  Thời gian/câu   {summary['seconds_per_question']}s")
        print(f"  Token           {self.usage.to_dict()}")
        if self.usage.num_context_at_risk:
            print(f"  CẢNH BÁO: {self.usage.num_context_at_risk} lời gọi có prompt dài hơn "
                  f"2 x {self.config.context_window} ký tự, có thể đã bị server cắt")
        print(f"\nĐã ghi kết quả vào {out_dir}/")
        return summary


def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        dataset=args.dataset,
        split=args.split,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        context_window=args.context_window,
        max_iterations=args.it,
        retrieval_top_k=args.top_k,
        num_candidate_triples=args.top_n,
        granularity_cascade=AB_TO_CASCADE[args.ab],
        use_triple_reranker=not args.no_reranker,
        triple_norm=args.triple_norm,
        official_rescue=args.official_rescue,
        min_source_docs=args.min_source_docs,
        passage_fallback=not args.no_passage_fallback,
        use_hybrid_retrieval=args.hybrid,
        force_openie_from_scratch=args.force_openie,
        num_workers=args.workers,
        embedding_model_name=args.embedding_name,
        embedding_max_length=args.embedding_max_length,
        openie_llm_name=args.openie_llm_name,
    )
    config.extra["candidate_scope"] = args.candidate_scope
    config.run_name = args.run_name or (
        f"{args.method}_{args.dataset}_{args.split}_it{args.it}"
        f"{'_ab-' + args.ab if args.ab else ''}"
        f"{'_noreranker' if args.no_reranker else ''}"
        f"{'_hybrid' if args.hybrid else ''}"
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy ViCiRAG trên dữ liệu tiếng Việt")
    parser.add_argument("--dataset", default="vimqa_sample")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--reason-lang", dest="reason_lang", default=None,
                        choices=["vi", "en", "en-vi"],
                        help="Ngôn ngữ prompt ICI/ACMG. Mặc định theo --lang. "
                             "en-vi = prompt tiếng Anh, buộc trả lời tiếng Việt.")
    parser.add_argument("--lang", default="vi", choices=["vi", "en"],
                        help="Ngôn ngữ prompt. Dùng en để tái lập trên bộ dữ liệu gốc.")
    parser.add_argument("--llm-name", dest="llm_name", default="google/gemini-2.5-flash")
    parser.add_argument("--openie-llm-name", dest="openie_llm_name",
                        default="google/gemini-2.5-flash-lite",
                        help="Model riêng cho trích triple hàng loạt (rẻ hơn)")
    parser.add_argument("--llm-base-url", dest="llm_base_url",
                        default="https://openrouter.ai/api/v1",
                        help="Server theo giao thức OpenAI. Chạy local bằng Ollama: "
                             "http://localhost:11434/v1 (khi đó không cần OPENROUTER_API_KEY)")
    parser.add_argument("--context-window", dest="context_window", type=int, default=None,
                        help="num_ctx của server local, để phát hiện prompt có thể bị cắt")
    parser.add_argument("--embedding-name", dest="embedding_name",
                        default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--embedding-max-length", dest="embedding_max_length", type=int,
                        default=1024,
                        help="Độ dài tối đa của encoder. bge-small-en-v1.5 chỉ nhận tới 512")
    parser.add_argument("--it", type=int, default=4, help="Số vòng ICI tối đa L")
    parser.add_argument("--top-k", dest="top_k", type=int, default=10, help="K đoạn/vòng")
    parser.add_argument("--top-n", dest="top_n", type=int, default=30, help="N triple ứng viên")
    parser.add_argument("--ab", default="", choices=sorted(AB_TO_CASCADE.keys()),
                        help="Ablation mức ngữ cảnh (Table 3)")
    parser.add_argument("--no-reranker", action="store_true", help="Ablation w/o reranker")
    parser.add_argument("--official-rescue", dest="official_rescue", action="store_true",
                        help="Cứu tập core triple như mã nguồn gốc: lấy ứng viên đầu khi rỗng, "
                             "và vớt triple được trích trong phần suy luận")
    parser.add_argument("--triple-norm", dest="triple_norm", default="none",
                        choices=["none", "official"],
                        help="official = chữ thường, bỏ ký tự đặc biệt như repo gốc "
                             "(chỉ cho tiếng Anh)")
    parser.add_argument("--no-passage-fallback", dest="no_passage_fallback",
                        action="store_true",
                        help="Bỏ mức dự phòng, lấy đúng hành vi Eq. 6 của bài báo gốc")
    parser.add_argument("--min-source-docs", dest="min_source_docs", type=int, default=0,
                        help="Số tài liệu tối thiểu tập triple lõi phải đến từ thì mức "
                             "ngữ cảnh mới được coi là đủ (0 = theo Eq. 6 gốc, 2 = đa bước)")
    parser.add_argument("--hybrid", action="store_true", help="Bật truy hồi lai BM25 + dense")
    parser.add_argument("--candidate-scope", dest="candidate_scope",
                        default="corpus", choices=["corpus", "sample"],
                        help="corpus = theo paper; sample = theo mã nguồn gốc")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu hỏi đầu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-openie", action="store_true")
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--method", default="cirag", choices=["cirag", "native", "ircot"],
                        help="cirag = phương pháp của paper; native/ircot = baseline")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass

    config = build_config_from_args(args)
    ViCiRAG(config, lang=args.lang, method=args.method,
            reason_lang=args.reason_lang).run(limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
