import { useState, useEffect, useRef } from 'react'

const TICKERS = ['AAPL','NVDA','MSFT','GOOGL','META','TSLA','AMZN','JPM','NFLX','AMD']

const ASCII_LOGO = `
 ███████╗ █████╗ ██████╗ ███╗   ██╗███╗   ███╗██╗
 ██╔════╝██╔══██╗██╔══██╗████╗  ██║████╗ ████║██║
 █████╗  ███████║██████╔╝██╔██╗ ██║██╔████╔██║██║
 ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║██║╚██╔╝██║██║
 ███████╗██║  ██║██║  ██║██║ ╚████║██║ ╚═╝ ██║███████╗
 ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝`

export default function SearchPage({ onSearch, error }) {
  const [ticker,     setTicker]     = useState('')
  const [useLstm,    setUseLstm]    = useState(true)
  const [useFinbert, setUseFinbert] = useState(false)
  const [useEnsemble,setUseEnsemble]= useState(true)
  const [focused,    setFocused]    = useState(false)
  const [time,       setTime]       = useState(new Date())
  const [bootLines,  setBootLines]  = useState([])
  const inputRef = useRef(null)

  // Clock
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  // Boot sequence
  const BOOT = [
    '> EARNML QUANTITATIVE INTELLIGENCE v2.0',
    '> Initialising ML inference engine…',
    '> Loading model registry: XGBoost, LightGBM, CatBoost, LSTM, FinBERT, GARCH',
    '> Connecting to Yahoo Finance data feeds…',
    '> Connecting to FRED macro API…',
    '> All systems nominal. Ready for analysis.',
    '> _',
  ]
  useEffect(() => {
    let i = 0
    const t = setInterval(() => {
      setBootLines(prev => [...prev, BOOT[i]])
      i++
      if (i >= BOOT.length) clearInterval(t)
    }, 220)
    return () => clearInterval(t)
  }, [])

  const handleSubmit = () => {
    if (ticker.trim()) onSearch({ ticker: ticker.trim().toUpperCase(), useLstm, useFinbert, useEnsemble })
  }

  return (
    <div className="min-h-screen grid-bg flex flex-col items-center justify-center px-4 py-12"
      style={{ background: 'var(--bg)' }}>

      {/* Top bar */}
      <div className="fixed top-0 left-0 right-0 h-8 border-b flex items-center justify-between px-4 z-50"
        style={{ background:'var(--bg)', borderColor:'var(--border)', fontSize:10 }}>
        <span style={{ color:'var(--green)' }}>EARNML SYSTEM</span>
        <span style={{ color:'var(--text-muted)' }}>
          {time.toUTCString().toUpperCase()}
        </span>
        <span style={{ color:'var(--green)' }}>◉ ONLINE</span>
      </div>

      <div className="w-full max-w-2xl mt-8">

        {/* ASCII logo */}
        <pre className="fade-up text-center mb-6 overflow-x-auto"
          style={{ fontSize:7, lineHeight:1.2, color:'var(--green)', opacity:0.7, letterSpacing:'0.05em' }}>
          {ASCII_LOGO}
        </pre>

        <div className="text-center mb-8 fade-up-1">
          <p className="label mb-2">Quantitative ML Earnings Intelligence</p>
          <p style={{ color:'var(--text-dim)', fontSize:11 }}>
            XGBoost · LightGBM · CatBoost · LSTM · FinBERT · GARCH · Macro
          </p>
        </div>

        {/* Boot terminal */}
        <div className="card corner mb-6 p-4 fade-up-2" style={{ minHeight:120 }}>
          <p className="label mb-3">SYSTEM LOG</p>
          {(bootLines || []).map((line, i) => (
            <p key={i} style={{
              fontSize:11, color: line?.includes('nominal') ? 'var(--green)' : 'var(--text-dim)',
              lineHeight:1.8,
            }}>{line}</p>
          ))}
        </div>

        {/* Ticker input */}
        <div className="card corner mb-4 p-5 fade-up-3">
          <p className="label mb-3">ENTER TICKER SYMBOL</p>
          <div className="flex items-center gap-3" style={{
            border: `1px solid ${focused ? 'var(--green)' : 'var(--border)'}`,
            background: focused ? 'rgba(0,200,150,0.04)' : 'transparent',
            padding:'12px 16px', borderRadius:2,
            boxShadow: focused ? '0 0 20px rgba(0,200,150,0.1)' : 'none',
            transition:'all 0.2s'
          }}>
            <span style={{ color:'var(--green)', fontSize:16, fontWeight:700 }}>$</span>
            <input
              ref={inputRef}
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              placeholder="AAPL"
              maxLength={5}
              style={{
                flex:1, background:'transparent', border:'none', outline:'none',
                color:'var(--text)', fontSize:28, fontWeight:700, fontFamily:'inherit',
                letterSpacing:'0.2em',
              }}
            />
            {focused && <span className="blink" style={{ color:'var(--green)', fontSize:28 }}>_</span>}
          </div>

          {/* Model toggles */}
          <div className="mt-4 grid grid-cols-3 gap-2">
            {([
                { key:'ensemble', label:'STACKED ENSEMBLE', sub:'XGB+LGB+CAT', val:useEnsemble, set:setUseEnsemble },
                { key:'lstm',    label:'LSTM PREDICTOR',    sub:'TensorFlow',  val:useLstm,    set:setUseLstm },
                { key:'finbert', label:'FINBERT NLP',       sub:'~440MB DL',   val:useFinbert, set:setUseFinbert },
              ]).map(m => (
                <button key={m.key} onClick={() => m.set(v => !v)}
                  style={{
                    border:`1px solid ${m.val ? 'var(--green)' : 'var(--border)'}`,
                    background: m.val ? 'rgba(0,200,150,0.06)' : 'transparent',
                    padding:'8px 10px', borderRadius:2, textAlign:'left', cursor:'pointer',
                    transition:'all 0.2s',
                  }}>
                  <p style={{ fontSize:9, letterSpacing:'0.15em', color: m.val ? 'var(--green)' : 'var(--text-muted)', marginBottom:2 }}>{m.label}</p>
                  <p style={{ fontSize:9, color:'var(--text-muted)' }}>{m.sub}</p>
                  <p style={{ fontSize:9, color: m.val ? 'var(--green)' : 'var(--text-muted)', marginTop:4 }}>
                    [{m.val ? '■ ON ' : '□ OFF'}]
                  </p>
                </button>
              ))}
          </div>

          <button onClick={handleSubmit} disabled={!ticker.trim()}
            className={ticker.trim() ? 'pulse-green' : ''}
            style={{
              marginTop:16, width:'100%', padding:'14px',
              background: ticker.trim() ? 'var(--green)' : 'transparent',
              border:`1px solid ${ticker.trim() ? 'var(--green)' : 'var(--border)'}`,
              color: ticker.trim() ? 'var(--bg)' : 'var(--text-muted)',
              fontSize:13, fontWeight:700, letterSpacing:'0.3em',
              fontFamily:'inherit', cursor: ticker.trim() ? 'pointer' : 'not-allowed',
              borderRadius:2, transition:'all 0.2s',
            }}>
            {ticker.trim() ? `▶ RUN ANALYSIS: ${ticker}` : '▶ ENTER TICKER TO PROCEED'}
          </button>
        </div>

        {error && (
          <div className="card p-4 mb-4 fade-up" style={{ borderColor:'rgba(255,77,109,0.4)', background:'rgba(255,77,109,0.05)' }}>
            <p className="label mb-2" style={{ color:'var(--red)' }}>ERROR</p>
            <pre style={{ fontSize:11, color:'var(--red)', whiteSpace:'pre-wrap', lineHeight:1.6 }}>{error}</pre>
          </div>
        )}

        {/* Quick tickers */}
        <div className="fade-up-4">
          <p className="label mb-3 text-center">QUICK ACCESS</p>
          <div className="flex flex-wrap justify-center gap-2">
            {TICKERS.map(t => (
              <button key={t} onClick={() => setTicker(t)}
                style={{
                  padding:'4px 12px', border:'1px solid var(--border)',
                  background:'transparent', color:'var(--text-dim)',
                  fontSize:11, fontFamily:'inherit', cursor:'pointer',
                  borderRadius:2, letterSpacing:'0.1em',
                  transition:'all 0.15s',
                }}
                onMouseEnter={e => { e.target.style.borderColor='var(--green)'; e.target.style.color='var(--green)' }}
                onMouseLeave={e => { e.target.style.borderColor='var(--border)'; e.target.style.color='var(--text-dim)' }}>
                {t}
              </button>
            ))}
          </div>
        </div>

        <p className="text-center mt-10 fade-up-5" style={{ fontSize:9, color:'var(--text-muted)', letterSpacing:'0.15em' }}>
          NOT FINANCIAL ADVICE · EDUCATIONAL ML PROJECT · FOR PORTFOLIO USE
        </p>
      </div>
    </div>
  )
}
