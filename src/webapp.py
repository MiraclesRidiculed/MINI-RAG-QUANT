from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from main import (
    _METRICS_PATH,
    build_arg_parser,
    build_runtime,
    predict_query,
    reference_kv_cache_config,
    set_seed,
)
from kv_cache import kv_cache_summary
from quantize import estimated_int8_symmetric_per_tensor_bytes, floating_point_weight_bytes


@dataclass(frozen=True)
class WebAppState:
    runtime: object
    metrics: dict[str, object]
    default_query: str


def parse_args() -> argparse.Namespace:
    defaults = build_arg_parser().parse_args([])
    parser = argparse.ArgumentParser(description="Serve the MINI-RAG-QUANT hackathon quantization UI in a browser.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8123, help="Port to serve the web UI on.")
    parser.add_argument("--query", type=str, default=defaults.query, help="Initial example query shown in the UI.")
    parser.add_argument("--retrain", action="store_true", help="Rebuild the dataset cache, tokenizer, and model before serving.")
    parser.add_argument("--skip-expanded-data", action="store_true", help="Skip the large normalized expansion dataset.")
    parser.add_argument("--expanded-train-per-label", type=int, default=defaults.expanded_train_per_label)
    parser.add_argument("--expanded-test-per-label", type=int, default=defaults.expanded_test_per_label)
    parser.add_argument("--classifier-weight", type=float, default=defaults.classifier_weight)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--differential-size", type=int, default=defaults.differential_size)
    parser.add_argument("--abstain-threshold", type=float, default=defaults.abstain_threshold)
    parser.add_argument("--margin-threshold", type=float, default=defaults.margin_threshold)
    parser.add_argument("--retrieval-threshold", type=float, default=defaults.retrieval_threshold)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def _load_metrics() -> dict[str, object]:
    if not _METRICS_PATH.exists():
        return {}
    return json.loads(_METRICS_PATH.read_text(encoding="utf-8"))


def _serialize_prediction(result: dict[str, object]) -> dict[str, object]:
    return {
        "assessment_sentence": result["assessment_sentence"],
        "primary_label": result["primary_label"],
        "primary_score": result["primary_score"],
        "best_model_label": result["best_model_label"],
        "best_model_score": result["best_model_score"],
        "unknown": result["unknown"],
        "low_confidence": result["low_confidence"],
        "open_set_flag": result["open_set_flag"],
        "top_margin": result["top_margin"],
        "top_retrieval_score": result["top_retrieval_score"],
        "normalized_entropy": result["normalized_entropy"],
        "differential": result["differential"],
        "ranked_predictions": [
            {"label": label, "score": score} for label, score in result["ranked_predictions"]
        ],
        "evidence_cards": result["evidence_cards"],
        "safety_findings": result["safety_findings"],
        "escalation_messages": result["escalation_messages"],
        "retrieved_cases": [
            {"score": item.score, "diagnosis": item.diagnosis, "symptoms": item.symptoms}
            for item in result["retrieved_cases"]
        ],
    }


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str, status: int = HTTPStatus.OK) -> None:
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def render_index(state: WebAppState) -> str:
    metrics = state.metrics or {}
    state_dict = state.runtime.model.state_dict()
    fp_bytes = floating_point_weight_bytes(state_dict)
    int8_bytes = estimated_int8_symmetric_per_tensor_bytes(state_dict)
    reduction_ratio = fp_bytes / max(int8_bytes, 1)
    kv_config = reference_kv_cache_config(state.runtime.model, sequence_length=512)
    kv_summary = kv_cache_summary(kv_config, window_size=128)
    bootstrap = {
        "defaultQuery": state.default_query,
        "metrics": metrics,
        "labelCount": len(state.runtime.labels),
        "trainExamples": len(state.runtime.dataset_splits["train"]),
        "testExamples": len(state.runtime.dataset_splits["test"]),
        "modelPath": str(Path("model_diagnosis.pth").resolve()),
        "fpBytes": fp_bytes,
        "int8Bytes": int8_bytes,
        "reductionRatio": reduction_ratio,
        "kvFp16Bytes": kv_summary["fp16_bytes"],
        "kvInt8Bytes": kv_summary["int8_bytes"],
        "kvSlidingBytes": kv_summary["sliding_window_fp16_bytes"],
        "kvInt8Ratio": kv_summary["int8_vs_fp16_ratio"],
        "kvSlidingRatio": kv_summary["sliding_vs_fp16_ratio"],
        "kvSeqLen": kv_config.sequence_length,
        "kvWindow": kv_summary["window_size"],
    }
    payload = json.dumps(bootstrap)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MINI-RAG-QUANT Quantization Demo</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --paper: rgba(255, 252, 247, 0.92);
      --ink: #13231f;
      --muted: #54635d;
      --accent: #8d2f23;
      --accent-soft: #f2d7ce;
      --teal: #1f5b57;
      --teal-soft: #d7ece8;
      --amber: #8a5c14;
      --amber-soft: #f6ead2;
      --line: rgba(19, 35, 31, 0.14);
      --shadow: 0 24px 60px rgba(49, 44, 34, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Aptos", "Trebuchet MS", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(141, 47, 35, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(31, 91, 87, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #efe5d6 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1200px, calc(100% - 32px));
      margin: 28px auto 48px;
    }}
    .hero {{
      display: grid;
      gap: 22px;
      grid-template-columns: 1.3fr 0.9fr;
      align-items: stretch;
    }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}
    .hero-copy {{
      padding: 28px 30px 26px;
      position: relative;
      overflow: hidden;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.74rem;
      color: var(--teal);
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-size: clamp(2.1rem, 4vw, 4rem);
      line-height: 0.95;
      max-width: 10ch;
    }}
    .hero-copy p {{
      margin: 18px 0 0;
      max-width: 58ch;
      font-size: 1rem;
      color: var(--muted);
      line-height: 1.6;
    }}
    .hero-stats {{
      display: grid;
      gap: 14px;
      padding: 24px;
      align-content: start;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .stat {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(19, 35, 31, 0.08);
    }}
    .stat-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .stat-value {{
      margin-top: 8px;
      font-size: 1.55rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .workspace {{
      display: grid;
      gap: 22px;
      margin-top: 22px;
      grid-template-columns: 0.92fr 1.08fr;
      align-items: start;
    }}
    .compose {{
      padding: 24px;
      position: sticky;
      top: 20px;
    }}
    .section-title {{
      margin: 0 0 14px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 1.45rem;
    }}
    textarea {{
      width: 100%;
      min-height: 180px;
      border-radius: 20px;
      border: 1px solid rgba(19, 35, 31, 0.16);
      background: rgba(255, 255, 255, 0.95);
      padding: 18px;
      font: inherit;
      color: var(--ink);
      resize: vertical;
      line-height: 1.5;
    }}
    .row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), #c04d38);
      color: white;
      padding: 13px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 14px 30px rgba(141, 47, 35, 0.22);
    }}
    .ghost {{
      background: rgba(255, 255, 255, 0.8);
      color: var(--teal);
      box-shadow: none;
      border: 1px solid rgba(31, 91, 87, 0.2);
    }}
    .chip {{
      border-radius: 999px;
      padding: 10px 14px;
      border: 1px solid rgba(19, 35, 31, 0.14);
      background: rgba(255, 255, 255, 0.72);
      cursor: pointer;
      color: var(--ink);
    }}
    .results {{
      display: grid;
      gap: 18px;
    }}
    .banner {{
      padding: 18px 20px;
      border-radius: 22px;
      border: 1px solid rgba(19, 35, 31, 0.1);
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.76));
    }}
    .banner.urgent {{
      background: linear-gradient(135deg, var(--accent-soft), rgba(255,255,255,0.88));
      border-color: rgba(141, 47, 35, 0.2);
    }}
    .banner.unknown {{
      background: linear-gradient(135deg, var(--amber-soft), rgba(255,255,255,0.9));
      border-color: rgba(138, 92, 20, 0.22);
    }}
    .banner.safe {{
      background: linear-gradient(135deg, var(--teal-soft), rgba(255,255,255,0.88));
      border-color: rgba(31, 91, 87, 0.16);
    }}
    .kicker {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .assessment {{
      margin-top: 10px;
      font-size: 1.1rem;
      line-height: 1.65;
    }}
    .tiles {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .card {{
      padding: 20px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: var(--paper);
      box-shadow: var(--shadow);
    }}
    .list {{
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }}
    .item {{
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(19, 35, 31, 0.09);
      background: rgba(255,255,255,0.78);
    }}
    .item strong {{
      display: inline-block;
      margin-right: 8px;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(31, 91, 87, 0.1);
      color: var(--teal);
      font-size: 0.88rem;
      font-weight: 700;
    }}
    .pill.warn {{
      background: rgba(138, 92, 20, 0.12);
      color: var(--amber);
    }}
    .pill.danger {{
      background: rgba(141, 47, 35, 0.12);
      color: var(--accent);
    }}
    a {{
      color: var(--accent);
    }}
    .metrics-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .metric-box {{
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(19, 35, 31, 0.08);
      background: rgba(255,255,255,0.74);
    }}
    .metric-box .label {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .metric-box .value {{
      margin-top: 8px;
      font-size: 1.15rem;
      font-weight: 700;
    }}
    .loading {{
      opacity: 0.72;
      pointer-events: none;
    }}
    .footer-note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.55;
    }}
    @media (max-width: 980px) {{
      .hero, .workspace, .tiles, .metrics-grid, .stat-grid {{
        grid-template-columns: 1fr;
      }}
      .compose {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-copy">
        <div class="eyebrow">Hackathon Quantization Demo</div>
        <h1>Show the memory drop before and after quantization.</h1>
        <p>
          The main point of this project is to demonstrate how a compact model can keep the same workflow
          while the weight payload becomes much smaller after quantization. The symptom-to-diagnosis flow is
          the demo workload that makes those memory numbers tangible and easy to present.
        </p>
      </div>
      <div class="panel hero-stats">
        <div class="section-title">Quantization + KV Cache Snapshot</div>
        <div class="stat-grid">
          <div class="stat"><div class="stat-label">FP32 Size</div><div class="stat-value" id="fpSize"></div></div>
          <div class="stat"><div class="stat-label">INT8 Estimate</div><div class="stat-value" id="int8Size"></div></div>
          <div class="stat"><div class="stat-label">KV Cache FP16</div><div class="stat-value" id="kvFp16Size"></div></div>
          <div class="stat"><div class="stat-label">KV Cache Optimized</div><div class="stat-value" id="kvOptimizedSize"></div></div>
          <div class="stat"><div class="stat-label">Model File</div><div class="stat-value" style="font-size:1rem;" id="modelPath"></div></div>
        </div>
      </div>
    </section>

    <section class="workspace">
      <aside class="panel compose" id="composeCard">
        <div class="section-title">Demo Input</div>
        <textarea id="queryInput" placeholder="Describe symptoms, timing, exposures, progression, and red flags."></textarea>
        <div class="row">
          <button id="runButton">Run Demo</button>
          <button class="ghost" id="resetButton" type="button">Reset Example</button>
        </div>
        <div class="row" id="exampleRow"></div>
        <div class="footer-note">
          Use this panel during the presentation to show that the same model pipeline can still produce an interpretable output while the main story remains memory efficiency and deployability.
        </div>
      </aside>

      <main class="results">
        <section class="panel card" id="summaryCard">
          <div class="section-title">Live Output</div>
          <div id="summaryBanner" class="banner">
            <div class="kicker">Waiting for query</div>
            <div class="assessment">Enter a demo query on the left to generate the formatted result.</div>
          </div>
          <div class="metrics-grid" id="diagnosticStrip"></div>
        </section>

        <section class="tiles">
          <div class="panel card">
            <div class="section-title">Quantization + KV Cache Story</div>
            <div class="list" id="memoryStory"></div>
          </div>
          <div class="panel card">
            <div class="section-title">Top Differential</div>
            <div class="list" id="differentialList"></div>
          </div>
        </section>

        <section class="tiles">
          <div class="panel card">
            <div class="section-title">Safety and Escalation</div>
            <div class="list" id="safetyList"></div>
          </div>
          <div class="panel card">
            <div class="section-title">Evidence and Retrieved Cases</div>
            <div class="list" id="evidenceList"></div>
            <div class="list" id="retrievalList"></div>
          </div>
        </section>

        <section class="panel card">
          <div class="section-title">Saved Benchmark Snapshot</div>
          <div class="metrics-grid" id="savedMetrics"></div>
          <div class="footer-note" id="criticalMetrics"></div>
        </section>
      </main>
    </section>
  </div>

  <script>
    const BOOT = {payload};
    const exampleQueries = [
      "fever and tingling at the bite site, then progress to confusion, hydrophobia (fear of water), paralysis",
      "I suddenly developed facial droop, slurred speech, and weakness in my left arm",
      "I have a fever from a wound infection and now I am confused, short of breath, clammy, and my heart is racing",
      "I missed my period, had a positive pregnancy test, and now I have lower abdominal pain with spotting"
    ];

    const els = {{
      queryInput: document.getElementById("queryInput"),
      runButton: document.getElementById("runButton"),
      resetButton: document.getElementById("resetButton"),
      composeCard: document.getElementById("composeCard"),
      exampleRow: document.getElementById("exampleRow"),
      fpSize: document.getElementById("fpSize"),
      int8Size: document.getElementById("int8Size"),
      kvFp16Size: document.getElementById("kvFp16Size"),
      kvOptimizedSize: document.getElementById("kvOptimizedSize"),
      modelPath: document.getElementById("modelPath"),
      summaryBanner: document.getElementById("summaryBanner"),
      diagnosticStrip: document.getElementById("diagnosticStrip"),
      memoryStory: document.getElementById("memoryStory"),
      differentialList: document.getElementById("differentialList"),
      safetyList: document.getElementById("safetyList"),
      evidenceList: document.getElementById("evidenceList"),
      retrievalList: document.getElementById("retrievalList"),
      savedMetrics: document.getElementById("savedMetrics"),
      criticalMetrics: document.getElementById("criticalMetrics"),
    }};

    function formatScore(value) {{
      return Number(value ?? 0).toFixed(3);
    }}

    function formatBytes(value) {{
      const units = ["B", "KiB", "MiB", "GiB"];
      let size = Number(value ?? 0);
      let unit = 0;
      while (size >= 1024 && unit < units.length - 1) {{
        size /= 1024;
        unit += 1;
      }}
      return `${{size.toFixed(size < 10 && unit > 0 ? 2 : 1)}} ${{units[unit]}}`;
    }}

    function statBox(label, value) {{
      return `<div class="metric-box"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
    }}

    function setStaticStats() {{
      els.fpSize.textContent = formatBytes(BOOT.fpBytes);
      els.int8Size.textContent = formatBytes(BOOT.int8Bytes);
      els.kvFp16Size.textContent = formatBytes(BOOT.kvFp16Bytes);
      els.kvOptimizedSize.textContent = formatBytes(BOOT.kvInt8Bytes);
      els.modelPath.textContent = BOOT.modelPath.split(/[/\\\\]/).slice(-1)[0];
      els.queryInput.value = BOOT.defaultQuery;
      exampleQueries.forEach((query) => {{
        const btn = document.createElement("button");
        btn.className = "chip";
        btn.type = "button";
        btn.textContent = query.length > 62 ? query.slice(0, 62) + "..." : query;
        btn.addEventListener("click", () => {{ els.queryInput.value = query; runPrediction(); }});
        els.exampleRow.appendChild(btn);
      }});
    }}

    function renderMemoryStory() {{
      const rows = [
        {{
          title: "Primary goal",
          body: "This hackathon demo is meant to show two things clearly: model weights shrink after quantization, and autoregressive inference memory can also shrink when the KV cache is optimized."
        }},
        {{
          title: "Before quantization",
          body: `Floating-point weights occupy about ${{formatBytes(BOOT.fpBytes)}}.`
        }},
        {{
          title: "After quantization",
          body: `The symmetric INT8 estimate is about ${{formatBytes(BOOT.int8Bytes)}}, which is roughly ${{BOOT.reductionRatio.toFixed(2)}}x smaller.`
        }},
        {{
          title: "KV cache baseline",
          body: `For a reference decoder at the same scale and a ${{BOOT.kvSeqLen}}-token sequence, the FP16 KV cache is about ${{formatBytes(BOOT.kvFp16Bytes)}}.`
        }},
        {{
          title: "KV cache optimization",
          body: `Quantized KV cache is about ${{formatBytes(BOOT.kvInt8Bytes)}} (~${{BOOT.kvInt8Ratio.toFixed(2)}}x smaller than FP16), and a sliding-window FP16 cache at ${{BOOT.kvWindow}} tokens is about ${{formatBytes(BOOT.kvSlidingBytes)}} (~${{BOOT.kvSlidingRatio.toFixed(2)}}x smaller).`
        }},
        {{
          title: "Why the diagnosis demo exists",
          body: `The symptom pipeline is the visible workload for the presentation: ${{BOOT.trainExamples.toLocaleString()}} train examples, ${{BOOT.testExamples.toLocaleString()}} test examples, and ${{BOOT.labelCount.toLocaleString()}} diagnoses.`
        }}
      ];
      els.memoryStory.innerHTML = rows.map((row) => `
        <div class="item">
          <strong>${{row.title}}</strong>
          <div class="footer-note">${{row.body}}</div>
        </div>
      `).join("");
    }}

    function renderSavedMetrics() {{
      const metrics = BOOT.metrics || {{}};
      const stack = metrics.inference_stack_metrics || {{}};
      const classifier = metrics.classifier_metrics || {{}};
      els.savedMetrics.innerHTML = [
        statBox("FP32 Size", formatBytes(BOOT.fpBytes)),
        statBox("INT8 Estimate", formatBytes(BOOT.int8Bytes)),
        statBox("Size Reduction", `${{BOOT.reductionRatio.toFixed(2)}}x`),
        statBox("KV Cache FP16", formatBytes(BOOT.kvFp16Bytes)),
        statBox("KV Cache INT8", formatBytes(BOOT.kvInt8Bytes)),
        statBox("KV Sliding FP16", formatBytes(BOOT.kvSlidingBytes)),
        statBox("Differential Top-3", formatScore(stack.differential_top3_recall)),
        statBox("Classifier Accuracy", formatScore(classifier.accuracy)),
        statBox("Retrieval Top-1", formatScore(stack.retrieval_top1_accuracy)),
        statBox("Diagnoses", (metrics.num_labels ?? BOOT.labelCount).toLocaleString()),
        statBox("Tokenizer Vocab", (metrics.tokenizer_vocab_size ?? 0).toLocaleString()),
      ].join("");

      const critical = metrics.critical_condition_metrics?.conditions || {{}};
      const rows = Object.entries(critical).map(([name, info]) =>
        `${{name}}: recall=${{formatScore(info.recall)}} (floor=${{formatScore(info.floor)}}, support=${{info.support}})`
      );
      els.criticalMetrics.textContent = rows.length
        ? `Critical benchmark: ${{rows.join(" | ")}}`
        : "Critical benchmark metrics will appear here after the CLI evaluation step runs.";
    }}

    function renderSummary(data) {{
      const urgent = data.safety_findings?.length > 0;
      const mode = data.unknown ? "unknown" : urgent ? "urgent" : "safe";
      const kicker = data.unknown
        ? "Unknown / clinician review"
        : urgent
          ? "Red-flag differential"
          : "Most likely diagnosis";
      const pill = data.unknown
        ? `<span class="pill warn">Rule out ${{data.primary_label}}</span>`
        : `<span class="pill ${{urgent ? "danger" : ""}}">${{data.primary_label}} | ${{formatScore(data.primary_score)}}</span>`;
      const confidence = `<span class="pill ${{data.low_confidence ? "warn" : ""}}">margin ${{formatScore(data.top_margin)}}</span>`;
      const retrieval = `<span class="pill">retrieval ${{formatScore(data.top_retrieval_score)}}</span>`;
      els.summaryBanner.className = `banner ${{mode}}`;
      els.summaryBanner.innerHTML = `
        <div class="kicker">${{kicker}}</div>
        <div class="assessment">${{data.assessment_sentence}}</div>
        <div class="row" style="margin-top:14px;">${{pill}}${{confidence}}${{retrieval}}</div>
      `;

      els.diagnosticStrip.innerHTML = [
        statBox("Primary", data.primary_label),
        statBox("Primary Score", formatScore(data.primary_score)),
        statBox("Unknown", data.unknown ? "Yes" : "No"),
        statBox("Margin", formatScore(data.top_margin)),
      ].join("");
    }}

    function renderDifferential(items) {{
      els.differentialList.innerHTML = items.length
        ? items.map((item) => `
            <div class="item">
              <strong>${{item.label}}</strong>
              <span class="pill">${{item.origin === "safety_rule" ? "safety rule" : "model ensemble"}}</span>
              <div class="meta">
                <span>score ${{formatScore(item.score)}}</span>
              </div>
              <div class="footer-note">${{item.rationale}}</div>
            </div>
          `).join("")
        : `<div class="item">No differential available.</div>`;
    }}

    function renderSafety(data) {{
      const findings = data.safety_findings || [];
      const escalations = data.escalation_messages || [];
      const parts = [];
      if (findings.length) {{
        findings.forEach((item) => parts.push(`
          <div class="item">
            <strong>${{item.condition}}</strong>
            <span class="pill danger">rule out urgently</span>
            <div class="meta"><span>rule score ${{formatScore(item.score)}}</span></div>
            <div class="footer-note">${{item.rationale}}</div>
          </div>
        `));
      }}
      if (escalations.length) {{
        escalations.forEach((message) => parts.push(`
          <div class="item">
            <strong>Escalation</strong>
            <div class="footer-note">${{message}}</div>
          </div>
        `));
      }}
      if (!parts.length) {{
        parts.push(`<div class="item">No safety-rule escalation triggered for this query.</div>`);
      }}
      els.safetyList.innerHTML = parts.join("");
    }}

    function renderEvidence(items) {{
      els.evidenceList.innerHTML = items.length
        ? items.map((card) => `
            <div class="item">
              <strong>${{card.condition}}</strong>
              <div class="footer-note">${{card.summary}}</div>
              <div class="footer-note">${{card.escalation}}</div>
              <div class="list" style="margin-top:10px;">
                ${{card.sources.map((source) => `
                  <div><a href="${{source.url}}" target="_blank" rel="noreferrer">${{source.title}}</a></div>
                `).join("")}}
              </div>
            </div>
          `).join("")
        : `<div class="item">No curated evidence cards for this result.</div>`;
    }}

    function renderRetrieval(items) {{
      els.retrievalList.innerHTML = items.length
        ? items.map((item) => `
            <div class="item">
              <strong>${{item.diagnosis}}</strong>
              <div class="meta"><span>similarity ${{formatScore(item.score)}}</span></div>
              <div class="footer-note">${{item.symptoms}}</div>
            </div>
          `).join("")
        : `<div class="item">No close training cases were retrieved.</div>`;
    }}

    async function runPrediction() {{
      const query = els.queryInput.value.trim();
      if (!query) return;
      els.composeCard.classList.add("loading");
      els.runButton.disabled = true;
      try {{
        const response = await fetch("/api/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ query }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Prediction failed");
        renderSummary(payload);
        renderDifferential(payload.differential || []);
        renderSafety(payload);
        renderEvidence(payload.evidence_cards || []);
        renderRetrieval(payload.retrieved_cases || []);
      }} catch (error) {{
        els.summaryBanner.className = "banner unknown";
        els.summaryBanner.innerHTML = `
          <div class="kicker">Request failed</div>
          <div class="assessment">${{error.message}}</div>
        `;
      }} finally {{
        els.composeCard.classList.remove("loading");
        els.runButton.disabled = false;
      }}
    }}

    els.runButton.addEventListener("click", runPrediction);
    els.resetButton.addEventListener("click", () => {{
      els.queryInput.value = BOOT.defaultQuery;
    }});
    els.queryInput.addEventListener("keydown", (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {{
        runPrediction();
      }}
    }});

    setStaticStats();
    renderMemoryStory();
    renderSavedMetrics();
    runPrediction();
  </script>
</body>
</html>"""


def build_handler(state: WebAppState):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                _html_response(self, render_index(state))
                return
            if parsed.path == "/api/metrics":
                _json_response(self, state.metrics)
                return
            if parsed.path == "/health":
                _json_response(self, {"status": "ok"})
                return
            _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/predict":
                _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload: dict[str, object] = {}
            if raw_body:
                if "application/json" in self.headers.get("Content-Type", ""):
                    payload = json.loads(raw_body)
                else:
                    payload = {key: values[0] for key, values in parse_qs(raw_body).items()}

            query = str(payload.get("query", "")).strip()
            if not query:
                _json_response(self, {"error": "Query is required."}, HTTPStatus.BAD_REQUEST)
                return

            result = predict_query(
                query=query,
                model=state.runtime.model,
                tokenizer=state.runtime.tokenizer,
                labels=state.runtime.labels,
                max_length=state.runtime.args.max_length,
                retriever=state.runtime.retriever,
                device=state.runtime.device,
                classifier_weight=state.runtime.args.classifier_weight,
                top_k=state.runtime.args.top_k,
                differential_size=state.runtime.args.differential_size,
                abstain_threshold=state.runtime.args.abstain_threshold,
                margin_threshold=state.runtime.args.margin_threshold,
                retrieval_threshold=state.runtime.args.retrieval_threshold,
            )
            _json_response(self, _serialize_prediction(result))

    return RequestHandler


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runtime = build_runtime(args)
    state = WebAppState(
        runtime=runtime,
        metrics=_load_metrics(),
        default_query=args.query,
    )

    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"Quantization demo UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
