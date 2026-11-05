"""Submission export checks; original plot encodings and data remain unchanged."""
import json
from pathlib import Path

import matplotlib
from matplotlib.text import Text


def export_figure(fig, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    assert abs(fig.get_figwidth()-6.5) < 1e-8, 'Use final manuscript width.'
    for text in fig.findobj(Text):
        if text.get_visible():
            text.set_fontsize(max(10, text.get_fontsize()))
            text.set_fontfamily('STIXGeneral')
            text.set_math_fontfamily('stix')
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    ticks, drawn_ticks = set(), set()
    for ax in fig.axes:
        if not ax.axison: continue
        for axis in [ax.xaxis, ax.yaxis]:
            lo, hi = sorted(axis.get_view_interval())
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                for text in [tick.label1, tick.label2]:
                    ticks.add(id(text))
                    if lo-1e-10 <= loc <= hi+1e-10: drawn_ticks.add(id(text))
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip(): continue
        if id(text) in ticks and id(text) not in drawn_ticks: continue
        if text.axes is not None and not text.axes.axison and text not in text.axes.texts:
            continue
        b = text.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or
                b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1):
            outside.append(text.get_text())
    if outside: raise RuntimeError(f'{out.name}: text outside canvas: {outside}')
    fig.savefig(out.with_suffix('.pdf'), dpi=450)
    fig.savefig(out.with_suffix('.png'), dpi=450)
    fonts = [text.get_fontsize() for text in fig.findobj(Text)
             if text.get_visible() and text.get_text().strip()]
    out.with_name(out.name+'_format.json').write_text(json.dumps({
        'width_in': float(fig.get_figwidth()), 'height_in': float(fig.get_figheight()),
        'minimum_base_text_pt': min(fonts), 'canvas_bounds': 'passed',
        'pdf_font_type': 42, 'png_dpi': 450,
        'font_family': 'STIXGeneral; STIX mathematics',
        'note': 'Empirical data and colors preserved; label and spacing revisions are documented in the renderer source.'
    }, indent=2)+'\n')
