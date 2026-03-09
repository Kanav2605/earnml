import { useState, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, RadialBarChart, RadialBar,
  PolarAngleAxis, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from 'recharts'

// ── Helpers ────────────────────────────────────────────────────────────────

const SIG = {
  STRONG_BUY:  { label:'STRONG BUY',  col:'#00ff9d', bg:'rgba(0,255,157,0.08)',  border:'rgba(0,255,157,0.35)' },
  BULLISH:     { label:'BULLISH',      col:'#00c896', bg:'rgba(0,200,150,0.08)',  border:'rgba(0,200,150,0.3)'  },
  NEUTRAL:     { label:'NEUTRAL',      col:'#94a3b8', bg:'rgba(148,163,184,0.06)',border:'rgba(148,163,184,0.2)'},
  BEARISH:     { label:'BEARISH',      col:'#ff4d6d', bg:'rgba(255,77,109,0.08)', border:'rgba(255,77,109,0.3)' },
  STRONG_SELL: { label:'STRONG SELL', col:'#ff1744', bg:'rgba(255,23,68,0.08)',  border:'rgba(255,23,68,0.35)' },
}
const sigFor = s => SIG[s] || SIG.NEUTRAL
const REGIME_COL = { LOW:'#00c896', NORMAL:'#f5a623', ELEVATED:'#f97316', EXTREME:'#ff4d6d' }
const CT = {
  contentStyle:{ background:'#07111c', border:'1px solid rgba(0,200,150,0.2)', borderRadius:2, fontFamily:'IBM Plex Mono', fontSize:10 },
  labelStyle:{ color:'#00c896' }, itemStyle:{ color:'#c8dbe8' },
}

function Panel({ label, tag, children, className='' }) {
  return (
    <div className={`card corner relative overflow-hidden ${className}`} style={{ padding:'20px' }}>
      <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:16 }}>
        <span style={{ fontSize:9, letterSpacing:'0.25em', color:'var(--text-muted)', textTransform:'uppercase' }}>{label}</span>
        {tag && <span style={{ fontSize:8, padding:'2px 6px', border:'1px solid var(--border)', color:'var(--text-muted)', letterSpacing:'0.1em' }}>{tag}</span>}
        <div style={{ flex:1, height:1, background:'linear-gradient(90deg,rgba(0,200,150,0.2),transparent)' }}/>
      </div>
      {children}
    </div>
  )
}

function Kpi({ label, value, sub, color, large }) {
  return (
    <div style={{ padding:'12px 14px', border:'1px solid var(--border)', background:'rgba(0,0,0,0.3)', borderRadius:2 }}>
      <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:4 }}>{label}</p>
      <p style={{ fontSize:large?26:17, fontWeight:700, color:color||'var(--text)', lineHeight:1, fontFamily:'IBM Plex Mono' }}>{value??'—'}</p>
      {sub && <p style={{ fontSize:9, color:'var(--text-muted)', marginTop:4 }}>{sub}</p>}
    </div>
  )
}

function StatusDot({ val }) {
  const c = val==='ok'?'var(--green)':val==='warn'?'var(--amber)':'var(--red)'
  return <span style={{ display:'inline-block', width:7, height:7, borderRadius:'50%', background:c, boxShadow:`0 0 5px ${c}`, marginRight:8, flexShrink:0 }}/>
}

const TABS = ['overview','commentary','classifier','sentiment','volatility','macro','backtest','history']

// ─────────────────────────────────────────────────────────────────────────────

export default function DashboardPage({ data, onReset }) {
  const [tab, setTab] = useState('overview')
  const [history, setHistory] = useState([])

  const { ticker, composite, models={}, macro={}, generated, elapsed_seconds, cached,
          commentary, backtest, drift } = data

  const clf  = models.classifier      || {}
  const lstm = models.price_predictor || {}
  const sent = models.sentiment       || {}
  const vol  = models.volatility      || {}

  const sig      = sigFor(composite?.signal)
  const score    = composite?.score ?? 0
  const scorePct = Math.round((score + 1) / 2 * 100)
  const beatProb = clf.beat_probability ?? 50
  const volChart = (vol.daily_forecast||[]).map((v,i)=>({d:`D${i+1}`,v}))
  const indiv    = clf.individual_models || {}
  const modelBars= Object.entries(indiv).map(([n,p])=>({name:n.substring(0,3).toUpperCase(),p}))

  // Load history when tab opened
  useEffect(() => {
    if (tab === 'history' && history.length === 0) {
      const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      fetch(`${base}/api/history/${ticker}`)
        .then(r => r.json())
        .then(d => setHistory(d.predictions || []))
        .catch(() => {})
    }
  }, [tab])

  return (
    <div className="min-h-screen grid-bg" style={{ background:'var(--bg)' }}>

      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <div style={{
        position:'sticky', top:0, zIndex:50,
        background:'rgba(3,5,10,0.94)', backdropFilter:'blur(12px)',
        borderBottom:'1px solid var(--border)',
        display:'flex', alignItems:'center', justifyContent:'space-between',
        padding:'0 20px', height:48, gap:12,
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <button onClick={onReset} style={{ color:'var(--text-muted)', fontSize:10, fontFamily:'inherit', background:'none', border:'none', cursor:'pointer', letterSpacing:'0.1em' }}>← NEW</button>
          <span style={{ color:'var(--border)' }}>|</span>
          <span style={{ color:'var(--green)', fontSize:18, fontWeight:800, letterSpacing:'0.3em', fontFamily:'Syne,sans-serif' }}>{ticker}</span>
          {drift?.retrain_recommended && (
            <span style={{ fontSize:8, padding:'2px 6px', border:'1px solid var(--amber)', color:'var(--amber)', letterSpacing:'0.1em', animation:'blink 1.5s step-end infinite' }}>⚠ DRIFT DETECTED</span>
          )}
          {cached && <span style={{ fontSize:8, padding:'2px 6px', border:'1px solid var(--border)', color:'var(--text-muted)', letterSpacing:'0.1em' }}>CACHED</span>}
        </div>

        <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              fontSize:8, letterSpacing:'0.2em', textTransform:'uppercase', fontFamily:'inherit',
              background:'none', border:'none', cursor:'pointer', padding:'4px 8px',
              color:      tab===t ? 'var(--green)' : 'var(--text-muted)',
              borderBottom: tab===t ? '1px solid var(--green)' : '1px solid transparent',
            }}>{t}</button>
          ))}
        </div>

        <div style={{ fontSize:9, color:'var(--text-muted)', whiteSpace:'nowrap' }}>
          {elapsed_seconds}s · {new Date(generated).toLocaleTimeString()}
        </div>
      </div>

      <div style={{ maxWidth:1400, margin:'0 auto', padding:'24px 20px 80px', display:'flex', flexDirection:'column', gap:16 }}>

        {/* ══ OVERVIEW ══════════════════════════════════════════════════ */}
        {tab==='overview' && (<>

          {/* Signal hero + gauge + lstm */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:16 }}>

            <Panel label="Composite ML Signal" tag={`${composite?.vote_count} MODEL VOTES`}>
              <div style={{ display:'inline-flex', alignItems:'center', gap:12, padding:'10px 18px', marginBottom:16,
                background:sig.bg, border:`1px solid ${sig.border}`, borderRadius:2 }}>
                <span style={{ fontSize:26, fontWeight:800, color:sig.col, fontFamily:'Syne,sans-serif', letterSpacing:'0.1em' }}>{sig.label}</span>
              </div>
              <div style={{ marginBottom:14 }}>
                <div style={{ display:'flex', justifyContent:'space-between', fontSize:9, color:'var(--text-muted)', marginBottom:5, letterSpacing:'0.1em' }}>
                  <span>-1.0 BEAR</span><span style={{color:sig.col}}>SCORE {score>0?'+':''}{score}</span><span>BULL +1.0</span>
                </div>
                <div style={{ height:6, background:'rgba(0,0,0,0.5)', border:'1px solid var(--border)', position:'relative', borderRadius:1 }}>
                  <div style={{ position:'absolute', left:'50%', top:0, bottom:0, width:1, background:'var(--border)' }}/>
                  <div style={{ position:'absolute', left:score>=0?'50%':`${50+score*50}%`, width:`${Math.abs(score)*50}%`,
                    top:0, bottom:0, background:score>0?'var(--green)':'var(--red)', boxShadow:`0 0 8px ${score>0?'var(--green)':'var(--red)'}` }}/>
                  <div style={{ position:'absolute', top:-4, bottom:-4, width:4, left:`${scorePct}%`, transform:'translateX(-50%)',
                    background:'white', borderRadius:1 }}/>
                </div>
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:6 }}>
                {(composite?.reasons||[]).map((r,i)=>(
                  <div key={i} style={{ fontSize:9, padding:'6px 8px', border:'1px solid var(--border)', color:'var(--text-dim)', lineHeight:1.6 }}>
                    <StatusDot val={composite.votes?.[i]>0?'ok':composite.votes?.[i]<0?'err':'warn'}/>
                    {r}
                  </div>
                ))}
              </div>
              {drift && (
                <div style={{ marginTop:12, padding:'8px 10px', border:`1px solid ${drift.status==='OK'?'var(--border)':'var(--amber)'}`,
                  background: drift.status==='OK'?'transparent':'rgba(245,166,35,0.05)', fontSize:9 }}>
                  <span style={{ color:'var(--text-muted)', marginRight:8 }}>MODEL HEALTH</span>
                  <span style={{ color: drift.status==='OK'?'var(--green)':'var(--amber)', fontWeight:700 }}>{drift.status}</span>
                  {drift.retrain_recommended && <span style={{ color:'var(--amber)', marginLeft:8 }}>— RETRAIN ADVISED</span>}
                </div>
              )}
            </Panel>

            <Panel label="Earnings Beat Probability" tag="STACKED ENSEMBLE">
              {'error' in clf ? <p style={{color:'var(--red)',fontSize:11}}>{clf.error}</p> : (<>
                <div style={{ height:155 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart cx="50%" cy="68%" innerRadius="55%" outerRadius="85%"
                      startAngle={180} endAngle={0} data={[{value:beatProb}]}>
                      <PolarAngleAxis type="number" domain={[0,100]} tick={false}/>
                      <RadialBar dataKey="value" cornerRadius={4}
                        background={{ fill:'rgba(0,200,150,0.06)' }}
                        fill={beatProb>57?'#00c896':beatProb<43?'#ff4d6d':'#f5a623'}/>
                      <text x="50%" y="56%" textAnchor="middle"
                        style={{ fontSize:32, fontWeight:800, fill:beatProb>57?'#00c896':beatProb<43?'#ff4d6d':'#f5a623', fontFamily:'IBM Plex Mono' }}>{beatProb}%</text>
                      <text x="50%" y="70%" textAnchor="middle"
                        style={{ fontSize:9, fill:'rgba(200,219,232,0.3)', letterSpacing:'0.15em' }}>BEAT PROB</text>
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:8 }}>
                  <Kpi label="PREDICTION" value={clf.prediction} color={clf.prediction==='UP'?'var(--green)':'var(--red)'}/>
                  <Kpi label="CONFIDENCE" value={`${clf.confidence}%`}/>
                  <Kpi label="FEATURES"   value={clf.features_used} sub="inputs"/>
                </div>
                {modelBars.length>0 && (
                  <div style={{ marginTop:12 }}>
                    <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:6 }}>BASE MODEL AGREEMENT</p>
                    <div style={{ height:55 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={modelBars} barSize={22}>
                          <XAxis dataKey="name" tick={{ fill:'rgba(200,219,232,0.4)', fontSize:9, fontFamily:'IBM Plex Mono' }}/>
                          <YAxis domain={[0,100]} hide/>
                          <Tooltip {...CT} formatter={v=>[`${v}%`,'Beat Prob']}/>
                          <Bar dataKey="p" radius={[2,2,0,0]}>
                            {modelBars.map((d,i)=>(<Cell key={i} fill={d.p>55?'var(--green)':d.p<45?'var(--red)':'var(--amber)'}/>))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </>)}
            </Panel>

            <Panel label="Price + Volatility" tag="LSTM · GARCH">
              {'predicted_return' in lstm && (
                <div style={{ marginBottom:14, paddingBottom:14, borderBottom:'1px solid var(--border)' }}>
                  <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:8 }}>LSTM {lstm.forward_days}D PREDICTION</p>
                  <div style={{ display:'flex', alignItems:'baseline', gap:10, marginBottom:8 }}>
                    <span style={{ fontSize:24, fontWeight:700, fontFamily:'IBM Plex Mono' }}>${lstm.current_price}</span>
                    <span style={{ color:'var(--text-muted)' }}>→</span>
                    <span style={{ fontSize:24, fontWeight:700, fontFamily:'IBM Plex Mono', color:lstm.predicted_return>0?'var(--green)':'var(--red)' }}>${lstm.predicted_price}</span>
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                    <Kpi label="RETURN" value={`${lstm.predicted_return>0?'+':''}${lstm.predicted_return}%`} color={lstm.predicted_return>0?'var(--green)':'var(--red)'}/>
                    <Kpi label="DIR. ACC" value={`${lstm.model_directional_acc}%`} sub="backtested"/>
                  </div>
                </div>
              )}
              {'regime' in vol && (<>
                <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:6 }}>GARCH REGIME</p>
                <p style={{ fontSize:22, fontWeight:800, fontFamily:'Syne,sans-serif', color:REGIME_COL[vol.regime], marginBottom:8 }}>{vol.regime}</p>
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                  <Kpi label="COND. VOL" value={`${vol.current_conditional_vol}%`} color="var(--amber)"/>
                  <Kpi label="VaR 99% 1D" value={`${vol.var_99_1d}%`} color="var(--red)"/>
                </div>
              </>)}
            </Panel>
          </div>

          {/* Sentiment + Macro */}
          <div style={{ display:'grid', gridTemplateColumns:'1.6fr 1fr', gap:16 }}>
            <Panel label="News Sentiment" tag="VADER + FINBERT">
              {'error' in sent ? <p style={{color:'var(--red)',fontSize:11}}>{sent.error}</p> : (<>
                <div style={{ display:'flex', justifyContent:'space-between', marginBottom:12 }}>
                  <div>
                    <p style={{ fontSize:26, fontWeight:800, fontFamily:'Syne,sans-serif',
                      color:sent.overall_sentiment==='BULLISH'?'var(--green)':sent.overall_sentiment==='BEARISH'?'var(--red)':'var(--text-muted)' }}>
                      {sent.overall_sentiment}
                    </p>
                    <p style={{ fontSize:10, color:'var(--text-muted)' }}>score {sent.sentiment_score>0?'+':''}{sent.sentiment_score} · {sent.model}</p>
                  </div>
                  <p style={{ fontSize:20, fontWeight:700, color:'var(--text)', fontFamily:'IBM Plex Mono' }}>{sent.headline_count} <span style={{fontSize:9,color:'var(--text-muted)'}}>headlines</span></p>
                </div>
                <div style={{ height:6, display:'flex', borderRadius:1, overflow:'hidden', marginBottom:5 }}>
                  <div style={{ background:'var(--green)', width:`${sent.finbert_bullish_pct||sent.vader_bullish_pct||0}%`, transition:'width 1s' }}/>
                  <div style={{ background:'rgba(148,163,184,0.2)', flex:1 }}/>
                  <div style={{ background:'var(--red)', width:`${sent.finbert_bearish_pct||sent.vader_bearish_pct||0}%`, transition:'width 1s' }}/>
                </div>
                <div style={{ display:'flex', justifyContent:'space-between', fontSize:9, color:'var(--text-muted)', marginBottom:12, letterSpacing:'0.1em' }}>
                  <span>▲ BULL {(sent.finbert_bullish_pct||sent.vader_bullish_pct||0).toFixed(0)}%</span>
                  <span>▼ BEAR {(sent.finbert_bearish_pct||sent.vader_bearish_pct||0).toFixed(0)}%</span>
                </div>
                <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                  {(sent.top_headlines||[]).slice(0,4).map((h,i)=>{
                    const pos=h.score>0.1, neg=h.score<-0.1
                    return (
                      <div key={i} style={{ display:'flex', gap:10, padding:'7px 10px', border:'1px solid var(--border)',
                        background:pos?'rgba(0,200,150,0.03)':neg?'rgba(255,77,109,0.03)':'transparent' }}>
                        <span style={{ fontSize:9, fontWeight:700, flexShrink:0, minWidth:38, color:pos?'var(--green)':neg?'var(--red)':'var(--text-muted)' }}>
                          {h.score>0?'+':''}{h.score?.toFixed(2)}
                        </span>
                        <p style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.5 }}>{h.title?.substring(0,88)}{h.title?.length>88?'…':''}</p>
                      </div>
                    )
                  })}
                </div>
              </>)}
            </Panel>

            <Panel label="Macro Context" tag="VIX · RATES · SPX">
              {macro && !macro.error ? (
                <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                  {[
                    { label:'VIX',           value:macro.vix,           badge:macro.vix_regime,   color:macro.vix_regime==='FEAR'?'var(--red)':'var(--green)' },
                    { label:'10Y RATE',       value:`${macro.t10y}%`,   badge:macro.rate_regime,  color:macro.rate_regime==='HIGH'?'var(--red)':'var(--green)' },
                    { label:'SPX 20D RET',   value:`${macro.spx_ret_20d>0?'+':''}${macro.spx_ret_20d}%`, color:macro.spx_ret_20d>0?'var(--green)':'var(--red)' },
                  ].map((m,i)=>(
                    <div key={i} style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'10px 12px', border:'1px solid var(--border)' }}>
                      <span style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)' }}>{m.label}</span>
                      <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                        {m.badge && <span style={{ fontSize:8, padding:'2px 5px', border:`1px solid ${m.color}`, color:m.color, letterSpacing:'0.1em' }}>{m.badge}</span>}
                        <span style={{ fontSize:18, fontWeight:700, color:m.color, fontFamily:'IBM Plex Mono' }}>{m.value}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : <p style={{color:'var(--text-muted)',fontSize:11}}>Macro unavailable</p>}
            </Panel>
          </div>

          {/* Vol chart */}
          {volChart.length>0 && (
            <Panel label="GARCH 10-Day Volatility Forecast" tag="ANN. %">
              <div style={{ height:130 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={volChart}>
                    <defs><linearGradient id="vg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--amber)" stopOpacity={0.25}/>
                      <stop offset="95%" stopColor="var(--amber)" stopOpacity={0}/>
                    </linearGradient></defs>
                    <XAxis dataKey="d" tick={{ fill:'rgba(200,219,232,0.3)', fontSize:9, fontFamily:'IBM Plex Mono' }}/>
                    <YAxis tick={{ fill:'rgba(200,219,232,0.3)', fontSize:9 }} tickFormatter={v=>`${v}%`}/>
                    <Tooltip {...CT} formatter={v=>[`${v}%`,'Ann. Vol']}/>
                    <ReferenceLine y={25} stroke="rgba(255,77,109,0.3)" strokeDasharray="4 4"/>
                    <Area type="monotone" dataKey="v" stroke="var(--amber)" fill="url(#vg)" strokeWidth={1.5}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:8, marginTop:10 }}>
                <Kpi label="5D FORECAST"   value={`${vol.garch_forecast_5d}%`}  color="var(--amber)"/>
                <Kpi label="10D FORECAST"  value={`${vol.garch_forecast_10d}%`} color="var(--amber)"/>
                <Kpi label="VaR 95% (1D)"  value={`${vol.var_95_1d}%`} color="var(--red)"/>
                <Kpi label="HIST. VOL 30D" value={`${vol.hist_vol_30d}%`}/>
              </div>
            </Panel>
          )}
        </>)}

        {/* ══ COMMENTARY ════════════════════════════════════════════════ */}
        {tab==='commentary' && (
          <Panel label="AI Analyst Commentary" tag="CLAUDE · INSTITUTIONAL STYLE">
            {!commentary ? (
              <p style={{color:'var(--text-muted)',fontSize:11}}>Commentary not generated. Re-run with use_llm=true.</p>
            ) : 'error' in commentary ? (
              <div>
                <p style={{color:'var(--amber)',fontSize:11,marginBottom:8}}>LLM unavailable — rule-based fallback:</p>
                {commentary.fallback && <CommentaryCard c={commentary.fallback}/>}
              </div>
            ) : <CommentaryCard c={commentary}/>}
          </Panel>
        )}

        {/* ══ BACKTEST ══════════════════════════════════════════════════ */}
        {tab==='backtest' && (
          <Panel label="Strategy Backtest" tag="vs BUY & HOLD">
            {!backtest ? (
              <div>
                <p style={{color:'var(--text-muted)',fontSize:11,marginBottom:12}}>Backtest not run. Re-run with run_backtest=true.</p>
                <p style={{fontSize:10,color:'var(--text-dim)',lineHeight:1.8}}>
                  The backtest simulates trading based on ML signals — buying when BULLISH, selling when BEARISH,
                  staying flat when NEUTRAL. It includes 10bp transaction costs per trade and
                  computes Sharpe, Sortino, max drawdown, and Calmar ratio vs buy-and-hold benchmark.
                </p>
              </div>
            ) : 'error' in backtest ? (
              <p style={{color:'var(--red)',fontSize:11}}>{backtest.error}</p>
            ) : (<>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:16 }}>
                <div>
                  <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--green)', marginBottom:10 }}>ML STRATEGY</p>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                    {Object.entries(backtest.ml||{}).map(([k,v])=>(
                      <Kpi key={k} label={k.replace(/_/g,' ').toUpperCase()} value={typeof v==='number'?v:String(v)}
                        color={k.includes('return')&&v>0?'var(--green)':k.includes('return')&&v<0?'var(--red)':undefined}/>
                    ))}
                  </div>
                </div>
                <div>
                  <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:10 }}>BUY & HOLD BENCHMARK</p>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                    {Object.entries(backtest.benchmark||{}).filter(([k])=>!['strategy','ticker'].includes(k)).map(([k,v])=>(
                      <Kpi key={k} label={k.replace(/_/g,' ').toUpperCase()} value={typeof v==='number'?v:String(v)}/>
                    ))}
                  </div>
                </div>
              </div>
              {backtest.trades && (
                <div style={{ padding:12, border:'1px solid var(--border)', background:'rgba(0,0,0,0.3)' }}>
                  <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:6 }}>TRADE SUMMARY</p>
                  <p style={{ fontSize:10, color:'var(--text-dim)' }}>
                    {backtest.trades.n_trades} trades · Win rate {backtest.trades.win_rate}% · Avg hold {backtest.trades.avg_hold_days} days
                  </p>
                </div>
              )}
            </>)}
          </Panel>
        )}

        {/* ══ HISTORY ═══════════════════════════════════════════════════ */}
        {tab==='history' && (
          <Panel label="Prediction History" tag="STORED IN SQLITE">
            {history.length === 0 ? (
              <p style={{color:'var(--text-muted)',fontSize:11}}>No history yet, or API not connected.</p>
            ) : (
              <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                {history.map((h,i)=>{
                  const s = sigFor(h.signal)
                  return (
                    <div key={i} style={{ display:'grid', gridTemplateColumns:'140px 100px 80px 80px 80px 1fr', gap:10, alignItems:'center', padding:'8px 12px', border:'1px solid var(--border)' }}>
                      <span style={{ fontSize:9, color:'var(--text-muted)' }}>{new Date(h.predicted_at).toLocaleString()}</span>
                      <span style={{ fontSize:10, fontWeight:700, color:s.col }}>{h.signal}</span>
                      <span style={{ fontSize:9, color:'var(--text-dim)' }}>BEAT {h.beat_probability?.toFixed(0)}%</span>
                      <span style={{ fontSize:9, color:h.predicted_return>0?'var(--green)':'var(--red)' }}>{h.predicted_return>0?'+':''}{h.predicted_return?.toFixed(1)}%</span>
                      <span style={{ fontSize:9, color:REGIME_COL[h.vol_regime]||'var(--text-muted)' }}>{h.vol_regime}</span>
                      <span style={{ fontSize:9, color:'var(--text-dim)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{h.llm_summary?.substring(0,60)||'—'}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </Panel>
        )}

        {/* ══ CLASSIFIER / SENTIMENT / VOLATILITY / MACRO ═══════════════ */}
        {tab==='classifier' && <ClassifierTab clf={clf} indiv={indiv}/>}
        {tab==='sentiment'  && <SentimentTab  sent={sent}/>}
        {tab==='volatility' && <VolatilityTab vol={vol} volChart={volChart}/>}
        {tab==='macro'      && <MacroTab macro={macro}/>}

        <div style={{ textAlign:'center', paddingTop:24 }}>
          <div style={{ height:1, background:'linear-gradient(90deg,transparent,var(--border),transparent)', marginBottom:12 }}/>
          <p style={{ fontSize:9, color:'var(--text-muted)', letterSpacing:'0.2em' }}>
            NOT FINANCIAL ADVICE · EARNML v3.0 · EDUCATIONAL ML PORTFOLIO PROJECT
          </p>
        </div>
      </div>
    </div>
  )
}

// ── Commentary card ──────────────────────────────────────────────

function CommentaryCard({ c }) {
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      <div style={{ display:'flex', gap:12, alignItems:'center', marginBottom:4 }}>
        <span style={{ fontSize:20, fontWeight:800, fontFamily:'Syne,sans-serif', color: c.recommendation==='BUY'?'var(--green)':c.recommendation==='SELL'?'var(--red)':'var(--amber)' }}>
          {c.recommendation}
        </span>
        <span style={{ fontSize:9, padding:'2px 8px', border:`1px solid ${c.conviction==='HIGH'?'var(--green)':'var(--amber)'}`, color:c.conviction==='HIGH'?'var(--green)':'var(--amber)', letterSpacing:'0.2em' }}>
          {c.conviction} CONVICTION
        </span>
        {c.model && <span style={{ fontSize:8, color:'var(--text-muted)', letterSpacing:'0.1em' }}>{c.model}</span>}
      </div>

      {c.one_liner && (
        <p style={{ fontSize:13, fontStyle:'italic', color:'var(--green)', lineHeight:1.6, borderLeft:'2px solid var(--green)', paddingLeft:12 }}>
          "{c.one_liner}"
        </p>
      )}

      {[
        { key:'summary',          label:'SUMMARY' },
        { key:'earnings_outlook', label:'EARNINGS OUTLOOK' },
        { key:'price_outlook',    label:'PRICE OUTLOOK' },
        { key:'sentiment_read',   label:'SENTIMENT' },
        { key:'risk_assessment',  label:'RISK' },
        { key:'macro_context',    label:'MACRO' },
      ].map(({key,label}) => c[key] && (
        <div key={key} style={{ padding:'10px 14px', border:'1px solid var(--border)', background:'rgba(0,0,0,0.3)' }}>
          <p style={{ fontSize:8, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:6 }}>{label}</p>
          <p style={{ fontSize:11, color:'var(--text-dim)', lineHeight:1.8 }}>{c[key]}</p>
        </div>
      ))}

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:4 }}>
        {c.key_risks?.length>0 && (
          <div style={{ padding:'10px 14px', border:'1px solid rgba(255,77,109,0.2)', background:'rgba(255,77,109,0.04)' }}>
            <p style={{ fontSize:8, letterSpacing:'0.2em', color:'var(--red)', marginBottom:8 }}>KEY RISKS</p>
            {c.key_risks.map((r,i)=><p key={i} style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.7, marginBottom:3 }}>▼ {r}</p>)}
          </div>
        )}
        {c.key_catalysts?.length>0 && (
          <div style={{ padding:'10px 14px', border:'1px solid rgba(0,200,150,0.2)', background:'rgba(0,200,150,0.04)' }}>
            <p style={{ fontSize:8, letterSpacing:'0.2em', color:'var(--green)', marginBottom:8 }}>KEY CATALYSTS</p>
            {c.key_catalysts.map((c2,i)=><p key={i} style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.7, marginBottom:3 }}>▲ {c2}</p>)}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Detail tabs ──────────────────────────────────────────────────

function ClassifierTab({ clf, indiv }) {
  return (
    <Panel label="Stacked Ensemble Classifier" tag="XGBoost + LightGBM + CatBoost → LogReg Meta-Learner">
      {'error' in clf ? <p style={{color:'var(--red)',fontSize:11}}>{clf.error}</p> : (
        <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }}>
            <Kpi label="BEAT PROB" value={`${clf.beat_probability}%`} large color={clf.beat_probability>55?'var(--green)':clf.beat_probability<45?'var(--red)':'var(--amber)'}/>
            <Kpi label="PREDICTION" value={clf.prediction} color={clf.prediction==='UP'?'var(--green)':'var(--red)'}/>
            <Kpi label="CONFIDENCE" value={`${clf.confidence}%`}/>
            <Kpi label="FEATURES" value={clf.features_used} sub="technical + macro + fundamental"/>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
            {[{n:'XGBoost',desc:'Gradient boosted trees. Handles feature interactions. Gain-based feature importance.'},
              {n:'LightGBM',desc:'Leaf-wise tree growth. Faster, better on high-dim data. Histogram algorithm.'},
              {n:'CatBoost',desc:'Symmetric trees with ordered boosting. Less tuning, handles small datasets.'},
            ].map(m=>(
              <div key={m.n} style={{ padding:14, border:'1px solid var(--border)', background:'rgba(0,0,0,0.3)' }}>
                <p style={{ fontSize:13, fontWeight:700, color:'var(--green)', marginBottom:8 }}>{m.n}</p>
                <p style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.7 }}>{m.desc}</p>
                {indiv[m.n.toLowerCase()] && <p style={{ fontSize:20, fontWeight:700, color:'var(--amber)', marginTop:10 }}>{indiv[m.n.toLowerCase()]}%</p>}
              </div>
            ))}
          </div>
          <div style={{ padding:14, border:'1px solid var(--green)', background:'rgba(0,200,150,0.04)' }}>
            <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--green)', marginBottom:8 }}>STACKING METHODOLOGY</p>
            <p style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.8 }}>
              Out-of-fold (OOF) predictions from each base model are used as features for the meta-learner —
              a Logistic Regression that learns the optimal combination. TimeSeriesSplit cross-validation
              prevents look-ahead bias. This is the standard approach in production quant ML systems
              and consistently outperforms any single model by 2–5% ROC-AUC.
            </p>
          </div>
        </div>
      )}
    </Panel>
  )
}

function SentimentTab({ sent }) {
  return (
    <Panel label="NLP Sentiment Analysis" tag="VADER + FinBERT Ensemble">
      {'error' in sent ? <p style={{color:'var(--red)',fontSize:11}}>{sent.error}</p> : (
        <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }}>
            <Kpi label="OVERALL" value={sent.overall_sentiment} large color={sent.overall_sentiment==='BULLISH'?'var(--green)':sent.overall_sentiment==='BEARISH'?'var(--red)':'var(--text-muted)'}/>
            <Kpi label="SCORE"       value={sent.sentiment_score} color={sent.sentiment_score>0?'var(--green)':'var(--red)'}/>
            <Kpi label="BULL %"      value={`${(sent.finbert_bullish_pct||0).toFixed(0)}%`} color="var(--green)"/>
            <Kpi label="BEAR %"      value={`${(sent.finbert_bearish_pct||0).toFixed(0)}%`} color="var(--red)"/>
          </div>
          {(sent.top_headlines||[]).map((h,i)=>(
            <div key={i} style={{ display:'flex', gap:12, padding:'10px 12px', border:'1px solid var(--border)',
              background:h.score>0.1?'rgba(0,200,150,0.03)':h.score<-0.1?'rgba(255,77,109,0.03)':'transparent' }}>
              <span style={{ fontSize:11, fontWeight:700, flexShrink:0, width:46, color:h.score>0.1?'var(--green)':h.score<-0.1?'var(--red)':'var(--text-muted)', fontFamily:'IBM Plex Mono' }}>
                {h.score>0?'+':''}{h.score?.toFixed(3)}
              </span>
              <p style={{ fontSize:11, color:'var(--text-dim)', lineHeight:1.6 }}>{h.title}</p>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

function VolatilityTab({ vol, volChart }) {
  const CT2 = { ...CT }
  return (
    <Panel label="GARCH(1,1) Volatility" tag="Student-t innovations">
      {'error' in vol ? <p style={{color:'var(--red)',fontSize:11}}>{vol.error}</p> : (
        <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }}>
            <Kpi label="REGIME"       value={vol.regime}      large color={REGIME_COL[vol.regime]}/>
            <Kpi label="COND. VOL"    value={`${vol.current_conditional_vol}%`} color="var(--amber)"/>
            <Kpi label="VaR 95% 1D"   value={`${vol.var_95_1d}%`}  color="var(--red)"/>
            <Kpi label="VaR 99% 1D"   value={`${vol.var_99_1d}%`}  color="var(--red)"/>
          </div>
          <div style={{ height:180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={volChart}>
                <XAxis dataKey="d" tick={{ fill:'rgba(200,219,232,0.35)', fontSize:10, fontFamily:'IBM Plex Mono' }}/>
                <YAxis tick={{ fill:'rgba(200,219,232,0.35)', fontSize:10 }} tickFormatter={v=>`${v}%`}/>
                <Tooltip {...CT2} formatter={v=>[`${v}%`,'Ann. Vol']}/>
                <ReferenceLine y={25} stroke="rgba(255,77,109,0.4)" strokeDasharray="4 2"/>
                <Bar dataKey="v" radius={[2,2,0,0]}>
                  {volChart.map((d,i)=>(<Cell key={i} fill={d.v>40?'var(--red)':d.v>25?'var(--amber)':'var(--green)'}/>))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ padding:12, border:'1px solid var(--border)', background:'rgba(0,0,0,0.3)' }}>
            <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--text-muted)', marginBottom:6 }}>WHAT IS GARCH?</p>
            <p style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.8 }}>
              GARCH models volatility clustering — large moves cluster together. Conditional variance
              is a function of past squared residuals (ARCH) and past variance (GARCH). Student-t
              innovations capture fat tails observed in financial returns. VaR quantifies max expected
              loss at a given confidence level over one trading day.
            </p>
          </div>
        </div>
      )}
    </Panel>
  )
}

function MacroTab({ macro }) {
  return (
    <Panel label="Macro Environment" tag="VIX · RATES · SPX">
      {macro && !macro.error ? (
        <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12 }}>
            <Kpi label="VIX"         value={macro.vix}          large color={macro.vix_regime==='FEAR'?'var(--red)':'var(--green)'} sub={macro.vix_regime}/>
            <Kpi label="10Y RATE"    value={`${macro.t10y}%`}   large color={macro.rate_regime==='HIGH'?'var(--red)':'var(--green)'} sub={macro.rate_regime}/>
            <Kpi label="SPX 20D RET" value={`${macro.spx_ret_20d>0?'+':''}${macro.spx_ret_20d}%`} large color={macro.spx_ret_20d>0?'var(--green)':'var(--red)'}/>
          </div>
          <div style={{ padding:14, border:'1px solid var(--green)', background:'rgba(0,200,150,0.03)' }}>
            <p style={{ fontSize:9, letterSpacing:'0.2em', color:'var(--green)', marginBottom:8 }}>REGIME-AWARE ML</p>
            <p style={{ fontSize:10, color:'var(--text-dim)', lineHeight:1.8 }}>
              Macro features (VIX, rate regime, SPX trend) are merged directly into the feature matrix
              used by the stacked ensemble. The same RSI reading in a VIX=12 vs VIX=35 environment
              is treated differently — this regime-aware feature engineering is standard in
              production quant systems at firms like JPMorgan.
            </p>
          </div>
        </div>
      ) : <p style={{color:'var(--text-muted)',fontSize:11}}>Macro data unavailable.</p>}
    </Panel>
  )
}
