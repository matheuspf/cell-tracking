"""Self-contained comparison, written only after all four new arms are scored."""
from __future__ import annotations

import json
from pathlib import Path

from .common import CONFIG, REPO, ROOT, RESULTS, read, sha, write

CANVAS=Path("/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/other-trackers-20260914.canvas.tsx")
LABELS={"v3":"v3 reference — training exposed", "ultrack_no_division":"Cellpose + ultrack, no divisions",
        "cellpose_general_v1_no_division_cuda":"Cellpose + HOCT general_v1, no divisions",
        "organoid":"Cellpose + OrganoidTracker2", "organoid_no_division":"Cellpose + OrganoidTracker2, no divisions",
        "cellect":"Cellpose + CELLECT association", "cellect_no_division":"Cellpose + CELLECT association, no divisions"}


def collect():
    summaries={};rows=[]
    previous=read(REPO/"results/hoct-reassessment-20260914/comparison.json")
    for item in previous["methods"]:
        if item["key"] in LABELS:summaries[item["key"]]=item["summaries"]
    rows.extend(r for r in previous["rows"] if r["arm"] in LABELS)
    for path in sorted(RESULTS.glob("summary-*.json")):
        result=read(path);assert result["config_sha256"]==sha(CONFIG)
        for key,value in result["summaries"].items():
            if key in summaries:assert summaries[key]==value
            summaries[key]=value
        rows.extend(result["rows"])
    assert set(summaries)==set(LABELS),"Both experiments must finish before the final report"
    unique={}
    for row in rows:
        key=row["arm"],row["dataset"]
        if key in unique:assert unique[key]["inputs"]==row["inputs"]
        unique[key]=row
    assert len(unique)==42
    return summaries,list(unique.values())


def run():
    summaries,rows=collect()
    current=["organoid","organoid_no_division","cellect","cellect_no_division"]
    best=max(current,key=lambda key:summaries[key]["pooled"]["score"])
    ul=summaries["ultrack_no_division"]["pooled"]["score"]
    s=summaries[best]["pooled"]
    facts=[]
    for family in ("organoid","cellect"):
        key=max((family,family+"_no_division"),key=lambda key:summaries[key]["pooled"]["score"])
        score=summaries[key]["pooled"]["score"]
        facts.append(f"{LABELS[key]} scores {score:.8f}, a {score-ul:+.8f} difference from the ultrack control.")
    facts.extend([
        "All comparisons use the same six complete 100-frame public clips. Each new family has a division-enabled arm and a no-division control; neither is fitted on Biohub labels.",
        "OrganoidTracker2 uses its released image-patch link and division probabilities with its pruned graph costs, solved exactly by HiGHS. It can omit Cellpose observations. Organoid-specific track cleanup and depth cutoffs are excluded.",
        "CELLECT's own image backbone supplies features at Cellpose centers. Its released between-frame matcher and gates are tested with those external observations. This is not an end-to-end run of CELLECT's native detector and grouping pipeline.",
        "These are reused development clips, not unseen-embryo validation. The v3 reference has known training overlap. Exact training lists for the CELLECT checkpoint are unresolved; OrganoidTracker's release describes mouse intestinal organoid training."
    ])
    org=summaries["organoid"]["pooled"]
    ul_summary=summaries["ultrack_no_division"]["pooled"]
    insight=(f"OrganoidTracker2's pooled gain is driven by division detection: its adjusted edge score is "
             f"{org['adj_edge_jaccard']:.8f}, below ultrack's {ul_summary['adj_edge_jaccard']:.8f}, "
             f"but its division reward adds {org['score']-org['adj_edge_jaccard']:.8f}. "
             "It improves on embryo 44b6 and regresses on 6bba. The small pooled gain is not a reliable new-embryo ranking.")
    error={}
    for key in current:
        subset=[r for r in rows if r["arm"]==key]
        error[key]={kind:{cause:sum(r.get(kind,{}).get(cause,0) for r in subset)
                         for cause in sorted({c for r in subset for c in r.get(kind,{})})}
                    for kind in ("missing_edges","false_edges")}
    methods=[dict(key=key,label=LABELS[key],summaries=summaries[key]) for key in LABELS]
    result=dict(best_new_arm=best,best_new_score=s["score"],delta_ultrack=s["score"]-ul,methods=methods,rows=rows,
                errors=error,facts=facts,insight=insight,config_sha256=sha(CONFIG),metric_revision=read(CONFIG)["metric_revision"])
    write(RESULTS/"comparison.json",result)
    doc=["# OrganoidTracker2 and CELLECT — 14 September 2026","",*sum(([f,""] for f in facts),[]),insight,"","## Full competition metric","",
         "Scores are exactly aggregated by the official metric, including node-count adjustment and division reward. They are not averages of per-clip scores.",""]
    for key in LABELS:
        p=summaries[key]["pooled"];c=p["counts"]
        doc.append(f"- **{LABELS[key]}:** {p['score']:.8f}; edges TP/FP/FN {c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']}; divisions {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {c['num_pred_nodes']:,} observations.")
    doc.extend(["","## Both embryos","", "This breakdown uses the fixed pooled configuration for each family; it does not select a different model on each embryo.",""])
    for family in ("organoid","cellect"):
        key=max((family,family+"_no_division"),key=lambda key:summaries[key]["pooled"]["score"])
        for embryo in ("44b6","6bba"):
            value=summaries[key][embryo]["score"];delta=value-summaries["ultrack_no_division"][embryo]["score"]
            doc.append(f"- {LABELS[key]}, embryo {embryo}: {value:.8f}; ultrack difference {delta:+.8f}.")
    doc.extend(["","## Error attribution","", "An unmatched prediction is not necessarily a false cell: annotations are sparse. False edges below are only those the official scorer considers evaluable.",""])
    for key in current:
        miss=error[key]["missing_edges"];false=error[key]["false_edges"]
        doc.append(f"- **{LABELS[key]}:** {miss.get('candidate_present_not_selected',0)} missed links had the correct candidate available; {miss.get('unmatched_endpoint',0)} lacked a matched endpoint; {miss.get('outside_candidate_bank',0)} were outside the candidate bank. Among evaluable false links, {false.get('touches_unmatched_endpoint',0)} touch an unmatched prediction and {false.get('both_matched_wrong_identity',0)} connect two matched cells with the wrong identities.")
    doc.extend(["","## Division controls",""])
    for family in ("organoid","cellect"):
        yes=summaries[family]["pooled"];no=summaries[family+"_no_division"]["pooled"];c=yes["counts"]
        doc.append(f"- {LABELS[family]}: enabling divisions changes the full score by {yes['score']-no['score']:+.8f}; {c['division_tp']} true positives, {c['division_fp']} evaluable false divisions, and {c['division_fn']} missed annotated divisions.")
    doc.extend(["","Only five divisions are annotated on this panel. These controls measure branching behavior here; they do not establish reliable mitosis generalization."])
    doc.extend(["", "The most useful next component to test is OrganoidTracker2's division probability with ultrack's stronger ordinary links. OrganoidTracker2 also loses 66 true links at its candidate-generation stage on this panel, so testing a less restrictive candidate bank is justified. Neither follow-up has been measured in this study."])
    doc.extend(["","## Verification and reproduction","", "OrganoidTracker's actual patch inputs and published calibration are compared to the released predictor. CPU/CUDA numerical checks use real image patches. The exact flow formulation is independently checked against every legal graph in 16 small cases. CELLECT feature-channel lookup is checked against the released expression, with border tile coverage and real-patch numerical checks. Every exported graph passes structural validation and a CSV round trip.",
                "", "The original sources and weights are pinned. No shared environment, raw input, incumbent graph or other study output was modified. This was a local runtime experiment; Kaggle notebook integration was not tested.",
                "", "- [Reproduction](../tools/other_trackers/README.md)","- [Complete counts](../results/other-trackers-20260914/comparison.json)",
                "- [Interactive comparison]("+str(CANVAS)+")", "- [OrganoidTracker source](https://github.com/jvzonlab/OrganoidTracker/tree/db28ff26584ac6d1230ea90750a3324a2778fb31)",
                "- [OrganoidTracker checkpoint release](https://zenodo.org/records/18479952)","- [CELLECT source](https://github.com/zzz333za/CELLECT/tree/3586070926f7f1fd5d8df37456861d22bdc63236)",""])
    (REPO/"docs/other-trackers-20260914.md").write_text("\n".join(doc))
    # Include only displayed summary fields so the artifact stays compact.
    display=[dict(key=m["key"],label=m["label"],summaries={g:dict(score=s["score"],counts=s["counts"]) for g,s in m["summaries"].items()}) for m in methods]
    payload=dict(methods=display,facts=facts,insight=insight,rows=[dict(arm=r["arm"],dataset=r["dataset"],missing_edges=r.get("missing_edges",{}),false_edges=r.get("false_edges",{})) for r in rows if r["arm"] in current])
    CANVAS.write_text(TEMPLATE.replace("__DATA__",json.dumps(payload,separators=(",",":"))))
    print(facts[:2],flush=True)


TEMPLATE='''import { BarChart, Button, Divider, Grid, H1, H2, Link, Row, Stack, Table, Text, useHostTheme, useState } from "cursor/canvas";
type Counts={edge_tp:number;edge_fp:number;edge_fn:number;division_tp:number;division_fp:number;division_fn:number;num_pred_nodes:number};
type Method={key:string;label:string;summaries:Record<string,{score:number;counts:Counts}>};
type ErrorRow={arm:string;dataset:string;missing_edges:Record<string,number>;false_edges:Record<string,number>};
const data:{methods:Method[];facts:string[];insight:string;rows:ErrorRow[]}=__DATA__;
const fmt=(n:number)=>n.toFixed(6);
const tri=(a:number,b:number,c:number)=>[a,b,c].map(n=>n.toLocaleString("en-US")).join(" / ");
export default function OtherTrackers(){
 const theme=useHostTheme();const [group,setGroup]=useState("pooled");const [arm,setArm]=useState("organoid");
 const selected=data.methods.find(m=>m.key===arm)!;
 const rows=data.rows.filter(r=>r.arm===arm&&(group==="pooled"||r.dataset.startsWith(group)));
 const missing=["unmatched_endpoint","outside_candidate_bank","candidate_present_not_selected"];
 const missingNames=["Endpoint not matched in output graph","Correct link outside candidate bank","Correct candidate available but unselected"];
 const best=(family:string)=>data.methods.filter(m=>m.key.startsWith(family)).sort((a,b)=>b.summaries.pooled.score-a.summaries.pooled.score)[0];
 const ul=data.methods.find(m=>m.key==="ultrack_no_division")!;
 return <Stack gap={24} style={{padding:24,maxWidth:1200,color:theme.text.primary}}>
  <Stack gap={8}><H1>OrganoidTracker2 and CELLECT on Cellpose observations</H1><Text>Full competition metric on six complete 100-frame clips, 14 September 2026.</Text><Text tone="secondary">Frozen released models; no local fitting. These public clips have been reused for development, and the v3 reference has known training overlap.</Text></Stack>
  <Stack gap={8}><H2>Result across all six clips</H2><Text>{data.insight}</Text></Stack>
  <Row gap={8} wrap>{[["pooled","All six clips"],["44b6","Embryo 44b6"],["6bba","Embryo 6bba"]].map(([key,label])=><Button key={key} variant={group===key?"primary":"secondary"} onClick={()=>setGroup(key)}>{label}</Button>)}</Row>
  <Grid columns="repeat(auto-fit,minmax(240px,1fr))" gap={24}>{[best("organoid"),best("cellect"),ul].map(m=><Stack key={m.key} gap={7}><Text tone="secondary">{m.label}</Text><H2>{fmt(m.summaries[group].score)}</H2><Text size="small">{m===ul?"Matched ultrack control":"Configuration selected on the pooled panel; selected cohort shown."}</Text></Stack>)}</Grid>
  <Divider/>
  <Stack gap={9}><H2>Full competition tracking score</H2><Text size="small" tone="secondary">Horizontal axis: score (unitless, higher is better). Vertical axis: configuration. Zero-based scale.</Text><BarChart horizontal height={340} yMin={0} yMax={1.1} categories={data.methods.map(m=>m.label.replace("Cellpose + ",""))} series={[{name:"Full competition score",tone:"info",data:data.methods.map(m=>m.summaries[group].score)}]} showValues/><Text size="small" tone="secondary">Source: pinned official scorer 075fc5f, all frames 0–99. Weighted edge aggregation plus division reward, including node-count adjustment.</Text>
   <div style={{overflowX:"auto"}}><Table headers={["Method","Score","Edge TP / FP / FN","Division TP / FP / FN","Observations"]} columnAlign={["left","right","right","right","right"]} rows={data.methods.map(m=>{const s=m.summaries[group],c=s.counts;return [m.label,fmt(s.score),tri(c.edge_tp,c.edge_fp,c.edge_fn),tri(c.division_tp,c.division_fp,c.division_fn),c.num_pred_nodes.toLocaleString("en-US")];})} striped/></div>
  </Stack>
  <Grid columns="repeat(auto-fit,minmax(340px,1fr))" gap={28}>
   <Stack gap={10}><H2>Missed-link diagnosis</H2><Row gap={6} wrap>{data.methods.filter(m=>m.key.startsWith("organoid")||m.key.startsWith("cellect")).map(m=><Button key={m.key} variant={arm===m.key?"primary":"secondary"} onClick={()=>setArm(m.key)}>{m.label.replace("Cellpose + ","")}</Button>)}</Row><Text>{selected.label}</Text><Table headers={["Failure stage","Missed true links"]} columnAlign={["left","right"]} rows={missing.map((k,i)=>[missingNames[i],rows.reduce((n,r)=>n+(r.missing_edges[k]||0),0).toLocaleString("en-US")])}/><Text size="small" tone="secondary">Post hoc diagnosis under official sparse-label matching. An unmatched endpoint does not by itself establish a false detection.</Text></Stack>
   <Stack gap={10}><H2>What each experiment measures</H2><Text>{data.facts[3]}</Text><Text>{data.facts[4]}</Text><Text size="small" tone="secondary">Candidate banks and observation selection differ. The comparison measures complete adapted pipelines, not an isolated classifier ablation.</Text></Stack>
  </Grid>
  <Divider/>
  <Stack gap={10}><H2>Verification and scope</H2><Text>Released preprocessing and calibration checks, real-patch CPU/CUDA comparisons, exact flow versus 16 exhaustive graph cases, structural validation and CSV round trips accompany the scores.</Text><Text>{data.facts[5]}</Text><Row gap={16} wrap><Link href="/home/mpf/code/kaggle/cell-tracking/docs/other-trackers-20260914.md">Results and interpretation</Link><Link href="/home/mpf/code/kaggle/cell-tracking/results/other-trackers-20260914/comparison.json">Exact counts</Link><Link href="/home/mpf/code/kaggle/cell-tracking/tools/other_trackers/README.md">Reproduction</Link></Row></Stack>
 </Stack>;
}
'''

if __name__=="__main__":run()
