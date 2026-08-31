"""Chart contract: static two-panel paired event dot plot for Word supplement.
Question: does the b=2 transfer direction persist across pressure-defined phases?
Each dot is one earthquake-level paired p90 NRMSE difference; 5 or 6 events.
Fixed original and boundary-retained policies; no connecting trend or uncertainty band.
Two categorical phases plus shaking; primary setting only, full sensitivities in CSV.
Blue circles/orange squares and direct legend; black median bars; white background.
1800x1060 SVG exported at 7 inches/1200 dpi by the shared renderer. Final QA in Word.
"""
from pathlib import Path
import pandas as pd
from make_figures import text,svg_open,INK,GRID
P=Path(__file__).resolve().parent
df=pd.read_csv(P/'phase_event_results.csv').query("smoothed==1 and eta==.5 and sampling_cap==120 and additional_stations_b==2 and metric=='profile_nrmse'")
svg=svg_open(1800,1060,'Phase-specific transfer of frozen three-station layouts',
    'Dots: earthquake-level paired differences; black bars: medians; negative values favour the transferred layout')
def dot(x,y,policy):
    if policy=='original':return f'<circle cx="{x}" cy="{y}" r="8" fill="#1769AA" stroke="white" stroke-width="1"/>'
    return f'<rect x="{x-7}" y="{y-7}" width="14" height="14" fill="#D97706" stroke="white" stroke-width="1"/>'
for x,policy,label in [(430,'original','Original layout'),(930,'boundary_retained','Both boundaries retained')]:
    svg += [dot(x,151,policy),text(x+24,160,label,29,weight=700)]
for panel,source in enumerate(['SKS02','SKS03']):
    left=170+panel*850;right=left+650;top=265;bottom=810
    target='SKS03' if source=='SKS02' else 'SKS02'
    y=lambda v:bottom-(v+.225)/.3*(bottom-top)
    svg.append(text(left,223,f'({chr(97+panel)}) {source} to {target}',32,weight=700))
    for tick in [-.20,-.15,-.10,-.05,0,.05]:
        yy=y(tick)
        svg += [f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="{INK if tick==0 else GRID}" stroke-width="{3 if tick==0 else 1.5}"/>',text(left-17,yy+9,f'{tick:+.2f}',28,anchor='end',weight=700)]
    for k,(phase,label) in enumerate([('shaking','Shaking'),('before_decay','Before decay'),('after_decay','After decay')]):
        xx=left+105+k*220
        n=0
        for j,policy in enumerate(['original','boundary_retained']):
            g=df[(df.source==source)&(df.phase==phase)&(df.policy==policy)].sort_values('event');n=len(g)
            centre=xx+(-37 if j==0 else 37)
            for order,v in enumerate(g.delta):
                assert -.225<=v<=.075
                svg.append(dot(centre+(order-(len(g)-1)/2)*8,y(v),policy))
            med=y(g.delta.median())
            svg.append(f'<line x1="{centre-28}" y1="{med}" x2="{centre+28}" y2="{med}" stroke="{INK}" stroke-width="4"/>')
        svg += [text(xx,858,label,28,anchor='middle',weight=700),text(xx,897,f'n = {n} earthquakes',25,anchor='middle',weight=600)]
svg.append('<text transform="translate(45,555) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="29" font-weight="700" fill="#172033">Candidate minus uniform p90 NRMSE</text>')
svg += [text(65,965,'Decay: first sustained 60-s interval below half the event pressure-proxy peak; not complete dissipation.',27,weight=600),text(65,1009,'Source: Sinha et al. (2020a, b). Same two centrifuge experiments; phase results are not independent trials.',27,weight=600),'</svg>']
(P/'Fig_phase_transfer.svg').write_text('\n'.join(svg),encoding='utf8')
