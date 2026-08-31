"""Figure contract: two faceted dot plots of paired pile-level median error.
Six source/budget groups, five piles per group. Static manuscript PNG/SVG.
White background; blue/orange source colours; five non-colour pile symbols.
"""
from pathlib import Path
import pandas as pd
from make_figures import text,svg_open,INK,GRID
ROOT=Path(__file__).resolve().parent

def marker(x,y,pile,c):
    if pile=='F1':return f'<circle cx="{x}" cy="{y}" r="10" fill="{c}" stroke="{INK}" stroke-width="2"/>'
    if pile=='F2':return f'<rect x="{x-10}" y="{y-10}" width="20" height="20" fill="white" stroke="{c}" stroke-width="4"/>'
    if pile=='T1':return f'<path d="M{x},{y-13} L{x-12},{y+10} L{x+12},{y+10} Z" fill="{c}" stroke="{INK}" stroke-width="2"/>'
    if pile=='T2':return f'<path d="M{x},{y-13} L{x-12},{y} L{x},{y+13} L{x+12},{y} Z" fill="white" stroke="{c}" stroke-width="4"/>'
    return f'<path d="M{x-10},{y-10} L{x+10},{y+10} M{x-10},{y+10} L{x+10},{y-10}" fill="none" stroke="{c}" stroke-width="5"/>'

def main():
    df=pd.read_csv(ROOT/'delft_field_pile_comparisons.csv').query("reference_pitch_m==.2 and boundary_rule=='original' and interpolator=='linear'")
    svg=svg_open(1800,1050,'Frozen layouts on digital static-field reference profiles',
        'Five Delft piles; paired median across numbered loading stages; negative values favour the transferred layout')
    for j,p in enumerate(['F1','F2','T1','T2','T3']):
        xx=510+j*155;svg+=[marker(xx,155,p,INK),text(xx+22,164,p,26,weight=600)]
    for panel,source in enumerate(['SKS02','SKS03']):
        left=180+panel*840;right=left+650;top=260;bottom=825
        y=lambda v: bottom-(v+.05)/.30*(bottom-top)
        svg.append(text(left,221,f'{source}-selected layout',30,weight=700))
        for tick in [-.05,0,.05,.10,.15,.20,.25]:
            yy=y(tick)
            svg.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="{INK if tick==0 else GRID}" stroke-width="{3 if tick==0 else 2}"/>')
            svg.append(text(left-18,yy+8,f'{tick:+.2f}',25,anchor='end',weight=600))
        for k,b in enumerate([2,3,4]):
            xx=left+100+k*230
            for j,pile in enumerate(['F1','F2','T1','T2','T3']):
                v=float(df.query('layout_source==@source and additional_stations_b==@b and pile==@pile').median_paired_delta.iloc[0])
                svg.append(marker(xx+(j-2)*17,y(v),pile,'#1769AA' if panel==0 else '#D97706'))
            svg.append(text(xx,bottom+43,f'b = {b}',28,anchor='middle',weight=700))
            svg.append(text(xx,bottom+77,f'{b+1} retained',24,anchor='middle',weight=600))
    svg.append('<text transform="translate(47,600) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="28" font-weight="700" fill="#172033">Candidate minus uniform NRMSE</text>')
    svg.append(text(80,966,'Reference grid: 0.20 m; source instrument resolution: 0.20 m. Five piles from one site, not 30 independent trials.',26,weight=600))
    svg.append(text(80,1012,'Source: Duffy et al. (2025). Author-processed force profiles; static loading does not validate liquefaction performance.',26,weight=600))
    svg.append('</svg>')
    (ROOT/'Fig06_Digital_Field_Transfer.svg').write_text('\n'.join(svg),encoding='utf-8')

if __name__=='__main__':main()
