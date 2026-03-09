import { useEffect, useState } from 'react'

const STEPS = [
  { id:'data',       label:'DATA PIPELINE',   msg:'Fetching OHLCV + macro from Yahoo/FRED…' },
  { id:'classifier', label:'STACKED ENSEMBLE',msg:'Running XGBoost + LightGBM + CatBoost…' },
  { id:'lstm',       label:'LSTM NETWORK',    msg:'Computing 60-day sequence prediction…' },
  { id:'sentiment',  label:'NLP SENTIMENT',   msg:'Scoring headlines with VADER + FinBERT…' },
  { id:'volatility', label:'GARCH MODEL',     msg:'Fitting GARCH(1,1)-t volatility model…' },
  { id:'macro',      label:'MACRO CONTEXT',   msg:'Reading VIX, rates, SPX regime…' },
  { id:'complete',   label:'COMPOSITE SIGNAL',msg:'Computing ensemble vote + signal…' },
]

export default function LoadingPage({ ticker = '…' }) {
  const [activeStep, setActiveStep] = useState(0)
  const [done,       setDone]       = useState([])
  const [logs,       setLogs]       = useState([`[${new Date().toISOString()}] Analysis started for ${ticker}`])
  const [elapsed,    setElapsed]    = useState(0)

  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => setElapsed(((Date.now() - start) / 1000).toFixed(1)), 100)

    let i = 0
    const advance = setInterval(() => {
      if (i > 0) setDone(d => [...d, STEPS[i-1].id])
      if (i < STEPS.length) {
        setActiveStep(i)
        setLogs(l => [...l.slice(-12), `[${new Date().toISOString()}] ${STEPS[i]?.msg || ''}`])
        i++
      } else {
        clearInterval(advance)
      }
    }, 1600)

    return () => { clearInterval(timer); clearInterval(advance) }
  }, [])

  const pct = Math.round((done.length / STEPS.length) * 100)

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center px-4"
      style={{ background:'var(--bg)' }}>

      <div className="w-full max-w-2xl">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <p className="label mb-1">ANALYSING</p>
            <p style={{ fontSize:36, fontWeight:800, color:'var(--green)', letterSpacing:'0.2em', fontFamily:'Syne, sans-serif' }}>
              {ticker}
            </p>
          </div>
          <div className="text-right">
            <p className="label mb-1">ELAPSED</p>
            <p style={{ fontSize:32, fontWeight:700, color:'var(--amber)', fontFamily:'IBM Plex Mono' }}>
              {elapsed}s
            </p>
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ height:2, background:'rgba(0,200,150,0.1)', marginBottom:24, borderRadius:1, overflow:'hidden' }}>
          <div style={{
            height:'100%', width:`${pct}%`,
            background:'linear-gradient(90deg,var(--green),#00ffb3)',
            transition:'width 0.8s ease',
            boxShadow:'0 0 8px var(--green)',
          }}/>
        </div>

        {/* Model pipeline steps */}
        <div className="card corner p-5 mb-5">
          <p className="label mb-4">MODEL PIPELINE</p>
          <div className="space-y-3">
            {STEPS.map((step, i) => {
              const isDone   = done.includes(step.id)
              const isActive = activeStep === i && !isDone
              const isPending= i > activeStep

              return (
                <div key={step.id} className="flex items-center gap-4" style={{
                  opacity: isPending ? 0.25 : 1,
                  transition:'opacity 0.3s',
                }}>
                  {/* Status dot */}
                  <div style={{
                    width:8, height:8, borderRadius:'50%', flexShrink:0,
                    background: isDone ? 'var(--green)' : isActive ? 'var(--amber)' : 'var(--border)',
                    boxShadow: isActive ? '0 0 8px var(--amber)' : isDone ? '0 0 6px var(--green)' : 'none',
                    animation: isActive ? 'pulse-green 1s ease infinite' : 'none',
                  }}/>

                  {/* Label */}
                  <span style={{
                    fontSize:9, letterSpacing:'0.2em', width:120, flexShrink:0,
                    color: isDone ? 'var(--green)' : isActive ? 'var(--amber)' : 'var(--text-muted)',
                  }}>{step.label}</span>

                  {/* Bar */}
                  <div style={{ flex:1, height:1, background:'rgba(0,200,150,0.08)', position:'relative', overflow:'hidden' }}>
                    {(isDone || isActive) && (
                      <div style={{
                        position:'absolute', inset:0,
                        background: isDone ? 'var(--green)' : 'var(--amber)',
                        transform: isDone ? 'none' : 'translateX(-60%)',
                        animation: isActive ? 'ticker-scroll 1.5s linear infinite' : 'none',
                        width: isDone ? '100%' : '200%',
                        opacity:0.6,
                      }}/>
                    )}
                  </div>

                  {/* Status */}
                  <span style={{
                    fontSize:9, width:48, textAlign:'right', flexShrink:0,
                    color: isDone ? 'var(--green)' : isActive ? 'var(--amber)' : 'var(--text-muted)',
                  }}>
                    {isDone ? '✓ DONE' : isActive ? 'RUNNING' : 'WAIT'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Log terminal */}
        <div className="card p-4" style={{ fontFamily:'IBM Plex Mono', fontSize:10 }}>
          <p className="label mb-3">SYSTEM LOG</p>
          <div style={{ height:100, overflowY:'auto' }}>
            {logs.map((l, i) => (
              <p key={i} style={{ color:'var(--text-dim)', lineHeight:1.8, marginBottom:1 }}>{l}</p>
            ))}
            <span className="blink" style={{ color:'var(--green)' }}>_</span>
          </div>
        </div>

        <p className="text-center mt-4" style={{ fontSize:10, color:'var(--text-muted)', letterSpacing:'0.15em' }}>
          {pct}% COMPLETE · This takes 30–90 seconds on first run
        </p>
      </div>
    </div>
  )
}
