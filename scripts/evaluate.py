"""Offline comparison of embedding models on the feed. No database needed.

Labels are proxies, and weak ones:
  - same industry = relevant (only companies whose industry is known)
  - cross-lingual: each Malay description should land nearest an English description of the same business concept

Usage: uv run python -m scripts.evaluate [--models a,b,c] [--out eval]
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from similar.embedding import build_embedder
from similar.fold import fold
from similar.normalize import QuarantineError, parse_message

FEED = Path("data/feed.jsonl")
K = 4
DEFAULT_MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  "hash-bow-384"]

# (English stem, Malay stem) of the same business concept, taken from the feed. Templated ones vary by city.
CONCEPTS = [
    ("Accounting, tax and company secretarial", "Perkhidmatan perakaunan, cukai"),
    ("Travel agency for umrah", "Agensi pelancongan umrah"),
    ("Freight forwarding and customs", "Agensi penghantaran fret"),
    ("Boutique hotel in", "Hotel butik di"),
    ("Civil engineering works", "Kerja-kerja kejuruteraan awam"),
    ("Community pharmacy", "Farmasi komuniti"),
    ("Durian orchard", "Kebun durian"),
    ("General practice clinic", "Klinik perubatan am"),
    ("General contractor for residential", "Kontraktor am bangunan"),
    ("Oil palm plantation", "Ladang kelapa sawit"),
    ("Last-mile parcel delivery", "Penghantaran bungkusan ke seluruh"),
    ("Mini market selling groceries", "Kedai runcit barangan"),
    ("Operating cafes and coffee shops", "Mengusahakan kedai kopi"),
    ("Software development for small business", "Pembangunan perisian perakaunan"),
    ("Manufacture of furniture from rubberwood", "Pembuatan perabot daripada kayu getah"),
    ("Bakery producing bread", "Pengeluaran roti dan kek"),
    ("Production of rubber gloves", "Pengeluaran sarung tangan getah"),
    ("Manufacture of plastic packaging", "Pengilangan bungkusan plastik"),
    ("Renovation and interior fit-out", "Pengubahsuaian dan kerja dalaman"),
    ("Homestay and chalet", "Pengusaha homestay dan chalet"),
    ("Retail of mobile phones", "Penjualan telefon bimbit"),
    ("Poultry farming", "Penternakan ayam"),
    ("Home nursing care", "Perkhidmatan jagaan warga emas"),
    ("Halal catering", "Perkhidmatan katering makanan halal"),
    ("Cold chain transport", "Perkhidmatan pengangkutan dan penyimpanan barangan sejuk beku"),
    ("Online retail of baby products", "Perniagaan dalam talian produk bayi"),
    ("Management consulting for SMEs", "Perunding pengurusan untuk PKS"),
    ("Tuition centre", "Pusat tuisyen"),
    ("Restaurant serving Malaysian", "Restoran masakan Melayu"),
    ("IT support and network installation", "Sokongan IT dan pemasangan rangkaian"),
    ("Private kindergarten and childcare", "Tadika swasta dan taska"),
]


def concept_of(text: str) -> tuple[int, str] | None:
    for i, (en, ms) in enumerate(CONCEPTS):
        if text.startswith(en):
            return i, "en"
        if text.startswith(ms):
            return i, "ms"
    return None


def load_companies():
    events, seen = [], set()
    for line in FEED.read_text(encoding="utf-8").splitlines():
        mid = hashlib.sha256(line.encode()).hexdigest()
        if mid in seen:
            continue
        seen.add(mid)
        try:
            events.append((mid, parse_message(line.encode())))
        except QuarantineError:
            pass
    return [c for c in fold(events).values() if c.status == "active" and c.nature_of_business]


def precision_ndcg(relevant: list[bool], total_relevant: int) -> tuple[float, float]:
    p = sum(relevant) / K
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(relevant))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(K, total_relevant)))
    return p, (dcg / idcg if idcg else 0.0)


def evaluate(model_name: str, companies) -> dict:
    texts = sorted({c.nature_of_business for c in companies})
    embedder = build_embedder(model_name, 384)
    start = time.perf_counter()
    vectors = np.array(embedder.embed(texts), dtype=np.float32)
    seconds = time.perf_counter() - start
    index = {t: i for i, t in enumerate(texts)}
    sims_text = vectors @ vectors.T

    # 1) Company level: what the portal would actually show (identical descriptions included).
    labelled = [c for c in companies if c.industry]
    ti = np.array([index[c.nature_of_business] for c in labelled])
    sims = sims_text[np.ix_(ti, ti)]
    np.fill_diagonal(sims, -np.inf)
    industries = np.array([c.industry for c in labelled])
    industry_sizes = Counter(industries.tolist())
    p_all, n_all = [], []
    for q in range(len(labelled)):
        # Stable tie-break by registration number, like the API.
        top = sorted(range(len(labelled)), key=lambda j: (-sims[q, j], labelled[j].registration_number))[:K]
        rel = [industries[j] == industries[q] for j in top]
        p, n = precision_ndcg(rel, industry_sizes[industries[q]] - 1)
        p_all.append(p)
        n_all.append(n)

    # 2) Unique-description level: identical texts removed, so the model has to generalise.
    majority = {}
    by_text = defaultdict(Counter)
    for c in labelled:
        by_text[c.nature_of_business][c.industry] += 1
    for t, counter in by_text.items():
        majority[t] = counter.most_common(1)[0][0]
    utexts = sorted(majority)
    ui = np.array([index[t] for t in utexts])
    usims = sims_text[np.ix_(ui, ui)]
    np.fill_diagonal(usims, -np.inf)
    ulabels = [majority[t] for t in utexts]
    usizes = Counter(ulabels)
    up, un, confusion, per_query = [], [], Counter(), []
    for q in range(len(utexts)):
        top = np.argsort(-usims[q], kind="stable")[:K]
        rel = [ulabels[j] == ulabels[q] for j in top]
        p, n = precision_ndcg(rel, usizes[ulabels[q]] - 1)
        up.append(p)
        un.append(n)
        for j in top:
            if ulabels[j] != ulabels[q]:
                confusion[(ulabels[q], ulabels[j])] += 1
        per_query.append((p, q, [(int(j), float(usims[q, j])) for j in top]))

    # 3) Cross-lingual: nearest English description to each Malay one, same concept?
    en = [(i, concept_of(t)[0]) for i, t in enumerate(texts) if concept_of(t) and concept_of(t)[1] == "en"]
    ms = [(i, concept_of(t)[0]) for i, t in enumerate(texts) if concept_of(t) and concept_of(t)[1] == "ms"]
    en_idx = np.array([i for i, _ in en])
    hits = 0
    for i, concept in ms:
        best = en_idx[int(np.argmax(sims_text[i, en_idx]))]
        hits += concept_of(texts[best])[0] == concept

    # Worst three unique descriptions (lowest P@4, then lowest nDCG), shown with what they were matched to.
    per_query.sort(key=lambda x: (x[0], un[x[1]], utexts[x[1]]))
    failures = [
        {"query": f"[{ulabels[q]}] {utexts[q]}",
         "precision@4": p,
         "neighbours": [f"[{ulabels[j]}] {utexts[j]} ({s:.3f})" for j, s in top]}
        for p, q, top in per_query[:3]
    ]
    return {
        "model": model_name,
        "embed_seconds": round(seconds, 2),
        "unique_texts": len(texts),
        "company_level": {"queries": len(labelled), f"precision@{K}": round(float(np.mean(p_all)), 4),
                          f"ndcg@{K}": round(float(np.mean(n_all)), 4)},
        "unique_description_level": {"queries": len(utexts), f"precision@{K}": round(float(np.mean(up)), 4),
                                     f"ndcg@{K}": round(float(np.mean(un)), 4)},
        "cross_lingual_concept_acc@1": {"queries": len(ms), "accuracy": round(hits / len(ms), 4)},
        "top_confusions": [{"query_industry": a, "neighbour_industry": b, "count": n}
                           for (a, b), n in confusion.most_common(8)],
        "worst_queries": failures,
    }


def to_markdown(results: list[dict]) -> str:
    lines = ["# Embedding model comparison", "",
             "Generated by `uv run python -m scripts.evaluate`. Relevance = same industry (weak proxy).", "",
             f"| Model | P@{K} company | nDCG@{K} company | P@{K} unique text | nDCG@{K} unique text "
             "| MS→EN concept acc@1 | Embed time |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        c, u = r["company_level"], r["unique_description_level"]
        lines.append(f"| `{r['model']}` | {c[f'precision@{K}']:.3f} | {c[f'ndcg@{K}']:.3f} | "
                     f"{u[f'precision@{K}']:.3f} | {u[f'ndcg@{K}']:.3f} | "
                     f"{r['cross_lingual_concept_acc@1']['accuracy']:.3f} | {r['embed_seconds']}s |")
    q = results[0]
    lines += ["", f"Company-level queries: {q['company_level']['queries']}. "
              f"Unique-description queries: {q['unique_description_level']['queries']}. "
              f"Malay descriptions in the cross-lingual check: {q['cross_lingual_concept_acc@1']['queries']}.", ""]
    for r in results:
        lines += [f"## `{r['model']}`", "", "Most frequent industry mix-ups in the top 4, unique-description level (query → neighbour):", ""]
        lines += [f"- {c['query_industry']} → {c['neighbour_industry']}: {c['count']}" for c in r["top_confusions"]]
        lines += ["", "Worst three unique descriptions:", ""]
        for f in r["worst_queries"]:
            lines.append(f"- **{f['query']}** (P@4 = {f['precision@4']:.2f})")
            lines += [f"  - {n}" for n in f["neighbours"]]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out", type=Path, default=Path("eval"))
    args = parser.parse_args()
    companies = load_companies()
    results = [evaluate(m, companies) for m in args.models.split(",")]
    args.out.mkdir(exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "results.md").write_text(to_markdown(results), encoding="utf-8")
    print((args.out / "results.md").read_text(encoding="utf-8").encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
