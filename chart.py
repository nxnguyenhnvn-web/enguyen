"""FireAnt-style interactive chart (TradingView Lightweight Charts v5) rendered as a Streamlit HTML component.

Indicators are computed in the browser, so toggling MA / EMA / BOLL / MACD / RSI / STOCH is instant.
"""
import json

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

LWC = "https://unpkg.com/lightweight-charts@5.0.8/dist/lightweight-charts.standalone.production.js"


def _bars(df: pd.DataFrame) -> list:
    d = df[["time", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return [{"time": t.strftime("%Y-%m-%d"), "open": round(float(o), 3), "high": round(float(h), 3),
             "low": round(float(l), 3), "close": round(float(c), 3), "volume": float(v)}
            for t, o, h, l, c, v in zip(pd.to_datetime(d["time"]), d["open"], d["high"], d["low"], d["close"],
                                        d["volume"])]


def _fc(fc, paths: bool) -> dict | None:
    if not fc:
        return None
    days = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in fc["dates"]]
    line = lambda arr: [{"time": t, "value": round(float(v), 3)} for t, v in zip(days, arr)]
    out = {"mean": line(fc["mean"]), "low": line(fc["low"]), "high": line(fc["high"]), "engine": fc["engine"]}
    if paths:
        out["paths"] = [line(p) for p in np.asarray(fc["paths"])[:12]]
    return out


def render(df: pd.DataFrame, symbol: str, fc: dict | None = None, paths: bool = False, height: int = 720,
           default_range: str = "1Y", key: str = "main"):
    payload = {"bars": _bars(df), "fc": _fc(fc, paths), "symbol": symbol, "range": default_range, "key": key}
    html = TEMPLATE.replace("__LWC__", LWC).replace("__DATA__", json.dumps(payload)).replace("__H__", str(height))
    components.html(html, height=height, scrolling=False)


TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<script src="__LWC__"></script>
<style>
:root{--bg:#11141c;--panel:#1b1f2a;--line:#262b38;--text:#d1d4dc;--muted:#787b86;--acc:#f26f21;
      --up:#00c076;--down:#ff3b4e;--ref:#f0b90b}
*{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--text);
  font:12px -apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
#bar{display:flex;flex-wrap:wrap;gap:4px;align-items:center;padding:6px 4px;border-bottom:1px solid var(--line)}
.grp{display:flex;gap:2px;margin-right:8px;align-items:center}
.lbl{color:var(--muted);margin:0 4px 0 2px}
button{background:transparent;color:var(--muted);border:1px solid transparent;border-radius:4px;padding:3px 8px;
  font:inherit;cursor:pointer} button:hover{background:var(--panel);color:var(--text)}
button.on{color:var(--acc);border-color:#3a2a1f;background:#22191200}
button.on{background:#2a1d14}
#gear{margin-left:auto}
#settings{display:none;position:absolute;right:6px;top:38px;z-index:20;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:10px;width:250px;box-shadow:0 6px 24px #0008}
#settings label{display:flex;justify-content:space-between;align-items:center;margin:5px 0;color:var(--muted)}
#settings input{width:110px;background:var(--bg);border:1px solid var(--line);color:var(--text);border-radius:4px;padding:3px 6px}
#wrap{position:relative} #chart{width:100%}
#legend{position:absolute;left:8px;top:6px;z-index:5;pointer-events:none;line-height:1.55;font-variant-numeric:tabular-nums}
#legend .sym{color:#e8eaed;font-weight:700;margin-right:6px} #legend span{margin-right:8px}
.sub{position:absolute;left:8px;z-index:5;pointer-events:none;font-variant-numeric:tabular-nums}
</style></head><body>
<div id="bar">
  <div class="grp" id="ranges"></div>
  <div class="grp"><span class="lbl">|</span><button data-type="candle">Nến</button><button data-type="line">Đường</button></div>
  <div class="grp" id="ovl"><span class="lbl">Chỉ báo</span></div>
  <div class="grp" id="subs"><span class="lbl">Khung phụ</span></div>
  <button id="gear" title="Indicator settings">⚙</button>
</div>
<div id="settings">
  <label>MA periods <input id="s_ma"></label>
  <label>EMA periods <input id="s_ema"></label>
  <label>BOLL (n, k) <input id="s_bb"></label>
  <label>RSI (n) <input id="s_rsi"></label>
  <label>MACD (f, s, sig) <input id="s_macd"></label>
  <label>STOCH (k, smooth, d) <input id="s_stoch"></label>
</div>
<div id="wrap"><div id="chart"></div><div id="legend"></div></div>
<script>
const D = __DATA__, H = __H__, LC = LightweightCharts;
const STORE = "enguyen-chart-v1";
const DEF = {type:"candle", ovl:{MA:true, EMA:false, BOLL:false, VOL:true, FC:true}, subs:{MACD:true, RSI:true, STOCH:false},
             ma:"20,50,200", ema:"12,26", bb:"20,2", rsi:"14", macd:"12,26,9", stoch:"14,3,3"};
let S; try { S = Object.assign({}, DEF, JSON.parse(localStorage.getItem(STORE) || "{}")); } catch(e) { S = {...DEF}; }
S.ovl = Object.assign({}, DEF.ovl, S.ovl); S.subs = Object.assign({}, DEF.subs, S.subs);
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch(e) {} };
const UP="#00c076", DOWN="#ff3b4e", PAL=["#f0b90b","#3b82f6","#d05cf2","#1fc8e6","#f97316","#22c55e"];
const bars = D.bars, closes = bars.map(b=>b.close), times = bars.map(b=>b.time);
const nums = s => String(s).split(/[ ,;]+/).map(Number).filter(x=>x>0);

// ---------- indicators
const sma=(a,n)=>{let o=[],s=0;for(let i=0;i<a.length;i++){s+=a[i];if(i>=n)s-=a[i-n];o.push(i>=n-1?s/n:null)}return o};
const ema=(a,n)=>{let o=[],k=2/(n+1),p=null;for(let i=0;i<a.length;i++){if(a[i]==null){o.push(null);continue}
  p=p==null?a[i]:a[i]*k+p*(1-k);o.push(i>=n-1?p:null)}return o};
const std=(a,n)=>a.map((_,i)=>{if(i<n-1)return null;let s=a.slice(i-n+1,i+1),m=s.reduce((x,y)=>x+y)/n;
  return Math.sqrt(s.reduce((x,y)=>x+(y-m)**2,0)/n)});
function rsi(a,n){let o=[null],g=0,l=0;for(let i=1;i<a.length;i++){const d=a[i]-a[i-1],u=Math.max(d,0),w=Math.max(-d,0);
  if(i<=n){g+=u/n;l+=w/n;o.push(i==n?100-100/(1+g/(l||1e-9)):null)}else{g=(g*(n-1)+u)/n;l=(l*(n-1)+w)/n;o.push(100-100/(1+g/(l||1e-9)))}}return o}
function stoch(n,sm,dn){const k=bars.map((b,i)=>{if(i<n-1)return null;let hh=-1e18,ll=1e18;for(let j=i-n+1;j<=i;j++){hh=Math.max(hh,bars[j].high);ll=Math.min(ll,bars[j].low)}
  return hh==ll?50:100*(b.close-ll)/(hh-ll)});const ks=smaN(k,sm);return [ks,smaN(ks,dn)]}
function smaN(a,n){return a.map((_,i)=>{if(i<n-1)return null;const s=a.slice(i-n+1,i+1);return s.some(x=>x==null)?null:s.reduce((x,y)=>x+y)/n})}
const ser=a=>a.map((v,i)=>v==null?{time:times[i]}:{time:times[i],value:+v.toFixed(4)});

// ---------- toolbar
const RANGES={"1M":21,"3M":63,"6M":126,"1Y":250,"3Y":750,"All":1e9};
const rg=document.getElementById("ranges");
Object.keys(RANGES).forEach(r=>{const b=document.createElement("button");b.textContent=r;b.onclick=()=>{setRange(r)};b.dataset.r=r;rg.appendChild(b)});
const OVL=[["MA","MA"],["EMA","EMA"],["BOLL","BOLL"],["VOL","Volume"]].concat(D.fc?[["FC","Dự báo"]]:[]);
const SUBS=[["MACD","MACD"],["RSI","RSI"],["STOCH","Stoch"]];
function chip(parent,obj,key,label){const b=document.createElement("button");b.textContent=label;b.className=obj[key]?"on":"";
  b.onclick=()=>{obj[key]=!obj[key];b.className=obj[key]?"on":"";save();build(true)};parent.appendChild(b)}
OVL.forEach(([k,l])=>chip(document.getElementById("ovl"),S.ovl,k,l));
SUBS.forEach(([k,l])=>chip(document.getElementById("subs"),S.subs,k,l));
document.querySelectorAll("[data-type]").forEach(b=>{b.className=S.type==b.dataset.type?"on":"";
  b.onclick=()=>{S.type=b.dataset.type;document.querySelectorAll("[data-type]").forEach(x=>x.className=S.type==x.dataset.type?"on":"");save();build(true)}});
const gear=document.getElementById("gear"),pop=document.getElementById("settings");
gear.onclick=()=>{pop.style.display=pop.style.display=="block"?"none":"block"};
[["s_ma","ma"],["s_ema","ema"],["s_bb","bb"],["s_rsi","rsi"],["s_macd","macd"],["s_stoch","stoch"]].forEach(([id,k])=>{
  const el=document.getElementById(id);el.value=S[k];el.onchange=()=>{S[k]=el.value;save();build(true)}});

// ---------- chart
let chart=null, legendFns=[], curRange=D.range;
const fmt=v=>v==null?"–":(+v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const fvol=v=>v>=1e6?(v/1e6).toFixed(2)+"M":v>=1e3?(v/1e3).toFixed(1)+"K":String(v);
function setRange(r){curRange=r;if(!chart)return;document.querySelectorAll("#ranges button").forEach(b=>b.className=b.dataset.r==r?"on":"");
  const n=Math.min(RANGES[r],bars.length), extra=(S.ovl.FC&&D.fc)?D.fc.mean.length+2:3;
  chart.timeScale().setVisibleLogicalRange({from:bars.length-n,to:bars.length-1+extra})}

function build(keepRange){
  const prev=chart&&keepRange?chart.timeScale().getVisibleLogicalRange():null;
  if(chart){chart.remove();chart=null}
  const el=document.getElementById("chart"), w=el.clientWidth;
  if(w<50){chart=null;return}  // hidden Streamlit tab: wait until visible (see ResizeObserver below)
  const nSub=SUBS.filter(s=>S.subs[s[0]]).length;
  const barH=document.getElementById("bar").offsetHeight, total=Math.max(300,H-barH-4);
  const subH=nSub?Math.max(60,Math.min(130,Math.floor(total*0.45/nSub))):0;  // price pane keeps >= 55%
  el.style.height=total+"px";
  chart=LC.createChart(el,{width:w,height:total,layout:{background:{color:"#11141c"},textColor:"#9aa0ab",fontSize:11,
      panes:{separatorColor:"#262b38",separatorHoverColor:"#3a4050"}},
    grid:{vertLines:{color:"#1a1e28"},horzLines:{color:"#1a1e28"}},
    crosshair:{mode:LC.CrosshairMode.Normal,vertLine:{color:"#4a5060",labelBackgroundColor:"#2a2f3c"},horzLine:{color:"#4a5060",labelBackgroundColor:"#2a2f3c"}},
    rightPriceScale:{borderColor:"#262b38"},timeScale:{borderColor:"#262b38",rightOffset:4},
    localization:{dateFormat:"dd/MM/yyyy",priceFormatter:p=>p.toFixed(2)}});
  legendFns=[];
  // price
  const main=S.type=="candle"
    ? chart.addSeries(LC.CandlestickSeries,{upColor:UP,downColor:DOWN,borderUpColor:UP,borderDownColor:DOWN,wickUpColor:UP,wickDownColor:DOWN})
    : chart.addSeries(LC.AreaSeries,{lineColor:"#f26f21",topColor:"rgba(242,111,33,.25)",bottomColor:"rgba(242,111,33,0)",lineWidth:2});
  main.setData(S.type=="candle"?bars:bars.map(b=>({time:b.time,value:b.close})));
  const over=[];
  if(S.ovl.MA) nums(S.ma).forEach((n,i)=>over.push(["MA"+n,sma(closes,n),PAL[i%PAL.length]]));
  if(S.ovl.EMA) nums(S.ema).forEach((n,i)=>over.push(["EMA"+n,ema(closes,n),PAL[(i+3)%PAL.length]]));
  if(S.ovl.BOLL){const [n,k]=nums(S.bb).concat([20,2]).slice(0,2),m=sma(closes,n),sd=std(closes,n);
    over.push(["BB up",m.map((v,i)=>v==null?null:v+k*sd[i]),"#787b86"],["BB mid",m,"#9aa0ab"],["BB dn",m.map((v,i)=>v==null?null:v-k*sd[i]),"#787b86"])}
  const overSeries=over.map(([name,arr,c])=>{const s=chart.addSeries(LC.LineSeries,{color:c,lineWidth:name.startsWith("BB")?1:1.5,
      lineStyle:name.startsWith("BB")&&name!="BB mid"?2:0,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
    s.setData(ser(arr));return [name,arr,c]});
  if(S.ovl.VOL){const v=chart.addSeries(LC.HistogramSeries,{priceScaleId:"vol",priceFormat:{type:"volume"},lastValueVisible:false,priceLineVisible:false});
    chart.priceScale("vol").applyOptions({scaleMargins:{top:.82,bottom:0}});
    v.setData(bars.map((b,i)=>({time:b.time,value:b.volume,color:(i&&b.close<bars[i-1].close?"rgba(255,59,78,.45)":"rgba(0,192,118,.45)")})))}
  if(S.ovl.FC&&D.fc){const fc=D.fc,link=[{time:bars[bars.length-1].time,value:bars[bars.length-1].close}];
    if(fc.paths) fc.paths.forEach(p=>chart.addSeries(LC.LineSeries,{color:"rgba(31,200,230,.25)",lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}).setData(link.concat(p)));
    [["high",2],["low",2],["mean",0]].forEach(([k,st])=>chart.addSeries(LC.LineSeries,{color:"#1fc8e6",lineWidth:k=="mean"?2:1,lineStyle:st,
      priceLineVisible:false,lastValueVisible:k=="mean",title:k=="mean"?"Dự báo":""}).setData(link.concat(fc[k])))}
  // sub panes
  let pane=0;
  const subLeg=[];
  if(S.subs.MACD){pane++;const [f,s,g]=nums(S.macd).concat([12,26,9]).slice(0,3),ef=ema(closes,f),es=ema(closes,s),m=ef.map((v,i)=>v==null||es[i]==null?null:v-es[i]);
    const sig=ema(m.map(x=>x==null?null:x),g),hist=m.map((v,i)=>v==null||sig[i]==null?null:v-sig[i]);
    chart.addSeries(LC.HistogramSeries,{priceLineVisible:false,lastValueVisible:false},pane).setData(hist.map((v,i)=>v==null?{time:times[i]}:{time:times[i],value:v,color:v>=0?"rgba(0,192,118,.6)":"rgba(255,59,78,.6)"}));
    chart.addSeries(LC.LineSeries,{color:"#3b82f6",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false},pane).setData(ser(m));
    chart.addSeries(LC.LineSeries,{color:"#f97316",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false},pane).setData(ser(sig));
    subLeg.push([pane,i=>`<b>MACD(${f},${s},${g})</b> <span style="color:#3b82f6">${fmt(m[i])}</span> <span style="color:#f97316">${fmt(sig[i])}</span> <span style="color:${hist[i]>=0?UP:DOWN}">${fmt(hist[i])}</span>`])}
  if(S.subs.RSI){pane++;const n=nums(S.rsi)[0]||14,r=rsi(closes,n);
    const rs=chart.addSeries(LC.LineSeries,{color:"#d05cf2",lineWidth:1.5,priceLineVisible:false,lastValueVisible:true},pane);rs.setData(ser(r));
    [70,30].forEach(l=>rs.createPriceLine({price:l,color:"#4a5060",lineStyle:2,lineWidth:1,axisLabelVisible:false}));
    subLeg.push([pane,i=>`<b>RSI(${n})</b> <span style="color:#d05cf2">${fmt(r[i])}</span>`])}
  if(S.subs.STOCH){pane++;const [n,sm,dn]=nums(S.stoch).concat([14,3,3]).slice(0,3),[k,d]=stoch(n,sm,dn);
    const ks=chart.addSeries(LC.LineSeries,{color:"#1fc8e6",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false},pane);ks.setData(ser(k));
    chart.addSeries(LC.LineSeries,{color:"#f0b90b",lineWidth:1.5,priceLineVisible:false,lastValueVisible:false},pane).setData(ser(d));
    [80,20].forEach(l=>ks.createPriceLine({price:l,color:"#4a5060",lineStyle:2,lineWidth:1,axisLabelVisible:false}));
    subLeg.push([pane,i=>`<b>Stoch(${n},${sm},${dn})</b> <span style="color:#1fc8e6">%K ${fmt(k[i])}</span> <span style="color:#f0b90b">%D ${fmt(d[i])}</span>`])}
  // stretch factors split the height: each sub-pane gets subH px, the price pane the rest
  const panes=chart.panes(), mainH=total-26-subH*(panes.length-1);
  panes[0].setStretchFactor(mainH); for(let p=1;p<panes.length;p++) panes[p].setStretchFactor(subH);
  // legends
  document.querySelectorAll(".sub").forEach(e=>e.remove());
  const subEls=subLeg.map(([p])=>{const e=document.createElement("div");e.className="sub";document.getElementById("wrap").appendChild(e);return e});
  const leg=document.getElementById("legend");
  function draw(i){const b=bars[i],pc=i?bars[i-1].close:b.open,ch=(b.close/pc-1)*100,c=ch>0?UP:ch<0?DOWN:"#f0b90b";
    const [y,mo,dd]=b.time.split("-");
    leg.innerHTML=`<span class="sym">${D.symbol}</span><span style="color:#9aa0ab">${dd}/${mo}/${y}</span>`+
      `<span>O <b style="color:${c}">${fmt(b.open)}</b></span><span>H <b style="color:${c}">${fmt(b.high)}</b></span>`+
      `<span>L <b style="color:${c}">${fmt(b.low)}</b></span><span>C <b style="color:${c}">${fmt(b.close)}</b></span>`+
      `<span style="color:${c}">${ch>=0?"+":""}${ch.toFixed(2)}%</span><span>KL <b>${fvol(b.volume)}</b></span><br>`+
      overSeries.filter(o=>!o[0].startsWith("BB")||o[0]=="BB mid").map(([n,a,col])=>`<span style="color:${col}">${n=="BB mid"?"BOLL("+S.bb+")":n} ${fmt(a[i])}</span>`).join("");
    // pane tops from pane heights (+1px separators); DOM rects are not laid out yet on first draw
    const hs=chart.panes().map(p=>p.getHeight()), tops=hs.map((_,k)=>hs.slice(0,k).reduce((a,b)=>a+b+1,0));
    subLeg.forEach(([p,f],j)=>{subEls[j].style.top=(tops[p]+4)+"px";subEls[j].innerHTML=f(i)})}
  chart.subscribeCrosshairMove(p=>{const i=p&&p.logical!=null?Math.max(0,Math.min(bars.length-1,Math.round(p.logical))):bars.length-1;draw(i)});
  if(prev) chart.timeScale().setVisibleLogicalRange(prev); else setRange(curRange);
  draw(bars.length-1); setTimeout(()=>draw(bars.length-1),50);  // again once panes have their heights
}
let lastW=0;
new ResizeObserver(()=>{const w=document.getElementById("chart").clientWidth;
  if(w<50||w==lastW)return; lastW=w;
  if(!chart)build(false); else chart.resize(w,chart.options().height||300)}).observe(document.body);
build(false);
</script></body></html>
"""
