"""Plotly figure builders for the Phase 6 risk report."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.reporting.report_builder import RiskReport


def build_risk_heatmap(report: RiskReport) -> go.Figure:
    """Build the per-category unguarded/guarded/delta heatmap."""
    categories = [
        item.owasp_category for item in report.per_category_risk.without_infrastructure_failures
    ]
    unguarded = [
        item.unguarded_risk_score
        for item in report.per_category_risk.without_infrastructure_failures
    ]
    guarded = [
        item.guarded_risk_score for item in report.per_category_risk.without_infrastructure_failures
    ]
    delta = [item.delta for item in report.per_category_risk.without_infrastructure_failures]
    z_values = [unguarded, guarded, delta]
    text = [
        [f"{value:.3f}" for value in unguarded],
        [f"{value:.3f}" for value in guarded],
        [f"{value:.3f}" for value in delta],
    ]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=categories,
                y=["Unguarded", "Guarded", "Delta"],
                colorscale="RdYlGn_r",
                zmid=0.0,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar={"title": "Risk"},
                hovertemplate="Category=%{x}<br>Series=%{y}<br>Value=%{z:.3f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Per-Category Risk Heatmap",
        height=420,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_xaxes(tickangle=-25)
    return figure


def build_layer_attribution_donut(report: RiskReport) -> go.Figure:
    """Build the layer attribution donut chart."""
    labels = [entry.label for entry in report.layer_attribution.entries]
    values = [entry.count for entry in report.layer_attribution.entries]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                sort=False,
                textinfo="label+percent",
                marker={
                    "colors": ["#1d4ed8", "#0f766e", "#f59e0b", "#7c3aed", "#dc2626", "#64748b"]
                },
            )
        ]
    )
    figure.update_layout(
        title="Layer Attribution",
        height=420,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return figure


def build_decision_distribution_bar(report: RiskReport) -> go.Figure:
    """Build the decision distribution bar chart."""
    labels = [entry.label for entry in report.decision_distribution.entries]
    values = [entry.count for entry in report.decision_distribution.entries]
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#2563eb",
                text=values,
                textposition="outside",
                hovertemplate="Decision=%{x}<br>Count=%{y}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Decision Distribution",
        height=420,
        margin={"l": 40, "r": 20, "t": 70, "b": 120},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_xaxes(tickangle=-25)
    return figure


def build_bypass_breakdown_bar(report: RiskReport) -> go.Figure:
    """Build the bypass-class stacked bar by OWASP category."""
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for attack in report.residual_analysis.residual_attacks:
        category_counts[attack.owasp_category][attack.bypass_class] += 1
    categories = sorted(category_counts)
    bypass_classes = ["A", "B", "C"]
    figure = go.Figure()
    colors = {"A": "#1d4ed8", "B": "#f97316", "C": "#7c3aed"}
    for bypass_class in bypass_classes:
        figure.add_trace(
            go.Bar(
                name=f"Bypass {bypass_class}",
                x=categories,
                y=[category_counts[category].get(bypass_class, 0) for category in categories],
                marker_color=colors[bypass_class],
                hovertemplate="Category=%{x}<br>Count=%{y}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Residual Bypass Breakdown by OWASP Category",
        barmode="stack",
        height=420,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


def build_cross_validator_agreement_bar(report: RiskReport) -> go.Figure:
    """Build the cross-validator agreement bar chart."""
    labels = [str(item["metric_name"]) for item in report.cross_validator_summary.metric_summaries]
    values = [
        _as_float(item["agreement_rate"])
        for item in report.cross_validator_summary.metric_summaries
    ]
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#0f766e",
                text=[f"{value:.0%}" for value in values],
                textposition="outside",
                hovertemplate="Metric=%{x}<br>Agreement=%{y:.1%}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Cross-Validator Agreement Rates",
        height=420,
        margin={"l": 40, "r": 20, "t": 70, "b": 80},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis={"range": [0.0, 1.05], "tickformat": ".0%"},
    )
    return figure


def build_owasp_web_chunking_chart(report: RiskReport) -> go.Figure:
    """Build the OWASP Web chunking finding chart with dual metrics."""
    finding = report.owasp_web_chunking_finding
    labels = ["contains_owasp_web", "nvd_only", "owasp_llm_only"]
    bullet_markers = [
        finding.contains_owasp_web_bullet_markers_per_chunk,
        finding.nvd_only_bullet_markers_per_chunk,
        finding.owasp_llm_only_bullet_markers_per_chunk,
    ]
    faithfulness = [
        finding.contains_owasp_web_mean_faithfulness,
        finding.nvd_only_mean_faithfulness,
        finding.owasp_llm_only_mean_faithfulness,
    ]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            name="Bullet Markers / Chunk",
            x=labels,
            y=bullet_markers,
            marker_color="#dc2626",
            hovertemplate="Slice=%{x}<br>Bullet markers=%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Bar(
            name="Mean Faithfulness",
            x=labels,
            y=faithfulness,
            marker_color="#2563eb",
            opacity=0.78,
            hovertemplate="Slice=%{x}<br>Faithfulness=%{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="OWASP Web Chunking Finding",
        barmode="group",
        height=420,
        margin={"l": 40, "r": 40, "t": 70, "b": 80},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_yaxes(title_text="Bullet Markers / Chunk", secondary_y=False)
    figure.update_yaxes(title_text="Mean Faithfulness", secondary_y=True, range=[0.0, 0.35])
    return figure


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
