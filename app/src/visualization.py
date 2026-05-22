"""Graphes Plotly."""
import plotly.graph_objects as go

def plot_survival_curve(sf, ttc_pred=None, title="Courbe de survie"):
    """Tracer une courbe de survie."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sf.x, y=sf.y,
        mode='lines',
        name='S(t)',
        line=dict(color='steelblue', width=3),
        hovertemplate='<b>t = %{x:.1f} min</b><br>S(t) = %{y:.3f}<extra></extra>'
    ))

    # Ligne médiane
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Médiane (S=0.5)",
        annotation_position="right"
    )

    # TTC prédit
    if ttc_pred is not None:
        fig.add_vline(
            x=ttc_pred,
            line_dash="dash",
            line_color="green",
            annotation_text=f"TTC = {ttc_pred:.1f} min",
            annotation_position="top"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Temps (min)",
        yaxis_title="S(t)",
        height=400,
        template='plotly_white',
        hovermode='x'
    )

    return fig
