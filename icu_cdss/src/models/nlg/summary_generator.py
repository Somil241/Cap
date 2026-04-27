"""Plain-English clinical summary renderer.

Pure-template fallback (no LLM) keyed off the aggregator payload. Handles
missing/None fields gracefully so we don't surface ``None`` to clinicians.
"""

from __future__ import annotations

from jinja2 import Template

REPORT = Template(
    """[{{ banner }}]
Patient {{ patient_id }} | {{ unit }} | {{ timestamp }}

SEPSIS  ({{ sepsis_prob }}, alert {{ sepsis_alert }})
  Key drivers: {{ key_drivers }}

VENTILATOR  ({{ vent_prob }} probability of need within 24 h)
  Predicted duration if initiated: ~{{ vent_dur }} (80% CI: {{ vent_ci_low }} – {{ vent_ci_high }})

LENGTH OF STAY
  Estimated remaining: ~{{ los_remaining }} (90% range: {{ los_p10 }} – {{ los_p90 }})

UNIT BED FORECAST ({{ unit }})
  24h: {{ bed_24 }}  |  48h: {{ bed_48 }}  |  72h: {{ bed_72 }}
"""
)


def _pct(x) -> str:
    return f"{round(float(x) * 100, 1)}%" if x is not None else "n/a"


def _hrs(x) -> str:
    return f"{round(float(x), 1)} h" if x is not None else "n/a"


def _beds(x) -> str:
    return f"{round(float(x), 1)}" if x is not None else "n/a"


def _banner(prob, alert_thr: float = 0.5, moderate_thr: float = 0.3) -> str:
    if prob is None:
        return "REPORT — DATA INCOMPLETE"
    if prob >= alert_thr:
        return "HIGH RISK — SEPSIS ALERT"
    if prob >= moderate_thr:
        return "MODERATE RISK"
    return "LOW RISK"


def generate_summary(payload: dict) -> str:
    sep = payload.get("sepsis", {}) or {}
    vent = payload.get("ventilator", {}) or {}
    los = payload.get("los", {}) or {}
    bed = payload.get("bed_demand", {}) or {}
    top = sep.get("top_features", []) or []

    drivers = (
        ", ".join(f"{x.get('name','?')} ({x.get('shap',0):+.2f})" for x in top[:3])
        if top else "n/a"
    )
    alert_thr = float(sep.get("alert_threshold", 0.5))
    moderate_thr = float(sep.get("moderate_threshold", 0.3))
    return REPORT.render(
        banner=_banner(sep.get("probability"), alert_thr, moderate_thr),
        patient_id=payload.get("patient_id", "unknown"),
        unit=bed.get("unit", "ICU"),
        timestamp=payload.get("timestamp", ""),
        sepsis_prob=_pct(sep.get("probability")),
        sepsis_alert=str(bool(sep.get("alert"))).lower(),
        key_drivers=drivers,
        vent_prob=_pct(vent.get("need_probability")),
        vent_dur=_hrs(vent.get("predicted_duration_hours")),
        vent_ci_low=_hrs(vent.get("ci_80", [None, None])[0]),
        vent_ci_high=_hrs(vent.get("ci_80", [None, None])[1]),
        los_remaining=_hrs(los.get("remaining_hours")),
        los_p10=_hrs(los.get("p10_hours")),
        los_p90=_hrs(los.get("p90_hours")),
        bed_24=_beds(bed.get("24h")),
        bed_48=_beds(bed.get("48h")),
        bed_72=_beds(bed.get("72h")),
    )
