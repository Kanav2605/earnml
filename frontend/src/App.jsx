import { useState } from 'react'
import SearchPage    from './pages/SearchPage.jsx'
import LoadingPage   from './pages/LoadingPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'

export default function App() {
  const [state,  setState]  = useState('search')
  const [data,   setData]   = useState(null)
  const [error,  setError]  = useState(null)
  const [ticker, setTicker] = useState('')

  const handleSearch = async ({ ticker: t, useLstm, useFinbert, useEnsemble }) => {
    setTicker(t); setState('loading'); setError(null)
    try {
      const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const res  = await fetch(`${base}/api/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ticker: t, use_lstm: useLstm, use_finbert: useFinbert, use_ensemble: useEnsemble, use_llm: true }),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || `API error ${res.status}`)
      setData(json); setState('dashboard')
    } catch (err) { setError(err.message); setState('search') }
  }

  if (state==='loading')             return <LoadingPage ticker={ticker}/>
  if (state==='dashboard' && data)   return <DashboardPage data={data} onReset={()=>{ setState('search'); setData(null) }}/>
  return <SearchPage onSearch={handleSearch} error={error}/>
}
