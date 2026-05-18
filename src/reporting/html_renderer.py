"""Standalone HTML renderer for the Phase 6 risk report."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from src.reporting.delta_visualizer import (
    build_bypass_breakdown_bar,
    build_cross_validator_agreement_bar,
    build_decision_distribution_bar,
    build_layer_attribution_donut,
    build_owasp_web_chunking_chart,
    build_risk_heatmap,
)
from src.reporting.report_builder import RiskReport


def render_risk_report_html(
    report: RiskReport,
    *,
    output_path: Path | str,
) -> Path:
    """Render one standalone risk report HTML file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = _environment().get_template("report.html.j2")
    charts = {
        "risk_heatmap": _chart_html(build_risk_heatmap(report), "risk-heatmap"),
        "layer_donut": _chart_html(build_layer_attribution_donut(report), "layer-donut"),
        "decision_bar": _chart_html(build_decision_distribution_bar(report), "decision-bar"),
        "bypass_bar": _chart_html(build_bypass_breakdown_bar(report), "bypass-bar"),
        "agreement_bar": _chart_html(build_cross_validator_agreement_bar(report), "agreement-bar"),
        "owasp_web_chart": _chart_html(build_owasp_web_chunking_chart(report), "owasp-web-chart"),
    }
    html = template.render(
        report=report,
        charts=charts,
        plotly_js=get_plotlyjs(),
    )
    output.write_text(html, encoding="utf-8")
    return output


def render_risk_report_html_from_json(
    *,
    risk_report_path: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Load one JSON report and render standalone HTML."""
    path = Path(risk_report_path)
    report = RiskReport.model_validate_json(path.read_text(encoding="utf-8"))
    html_path = Path(output_path) if output_path is not None else path.with_suffix(".html")
    return render_risk_report_html(report, output_path=html_path)


def _environment() -> Environment:
    templates_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _chart_html(figure: object, div_id: str) -> str:
    return cast(
        "str",
        to_html(
            figure,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={
                "displayModeBar": False,
                "responsive": True,
                "staticPlot": False,
            },
        ),
    )
