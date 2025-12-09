import { useState } from 'react';
import { submitPipeline } from './api';
import './styles.css';

function ResultCard({ title, children }) {
  return (
    <div className="result-card">
      <p className="label">{title}</p>
      {children}
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [context, setContext] = useState('');
  const [runTraining, setRunTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please choose a dataset to upload.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setLogs('');

    try {
      const payload = await submitPipeline({
        file,
        targetColumn: target,
        context,
        runTraining,
        apiBase: import.meta.env.VITE_API_BASE,
      });
      setResult(payload);
      setLogs(payload.logs || '');
    } catch (err) {
      setError(err.message || 'Something went wrong.');
      if (err.response?.logs) {
        setLogs(err.response.logs);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="glow" />
      <div className="glow glow-2" />
      <header className="hero">
        <p className="eyebrow">MLBot • Python + Gemini</p>
        <h1>Drop a dataset, describe it, and get an instant model plan.</h1>
        <p className="lede">
          The MERN wrapper sends your file, context, and target column straight into the Python orchestrator in
          <code> Smallest_Prototype/feature.py</code>.
        </p>
      </header>

      <main className="layout">
        <section className="panel form-panel">
          <div className="panel-head">
            <h2>Dataset Intake</h2>
            <p>We compute stats, ask the LLM for a task + model, and optionally kick off training.</p>
          </div>
          <form className="form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Dataset file</span>
              <div className="upload">
                <input
                  type="file"
                  accept=".csv,.tsv,.txt,.json,.xlsx,.xls"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
                <p className="hint">CSV/TSV/XLSX/JSON supported. File is sent to the API and piped into the Python script.</p>
              </div>
            </label>

            <label className="field">
              <span>Target column (optional)</span>
              <input
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="e.g. churn, price, outcome"
              />
            </label>

            <label className="field">
              <span>Dataset context / goal</span>
              <textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Who is this data about? What do you want to predict?"
                rows={4}
              />
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={runTraining}
                onChange={(e) => setRunTraining(e.target.checked)}
              />
              <span>Also run the suggested training script (may be interactive; best left off for API calls)</span>
            </label>

            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Running pipeline…' : 'Send to pipeline'}
            </button>

            {error && <p className="error">{error}</p>}
          </form>
        </section>

        <section className="panel results-panel">
          <div className="panel-head">
            <h2>Pipeline Output</h2>
            <p>Everything the Python orchestrator returned, decoded for the browser.</p>
          </div>

          {!result && !loading && (
            <div className="placeholder">Results will show up here after you submit.</div>
          )}

          {loading && <div className="loader">Crunching stats and asking the model…</div>}

          {result && (
            <div className="results-grid">
              <ResultCard title="Status">
                <p className={`pill pill-${result.status}`}>{result.status}</p>
                <p className="subtext">Exit code: {result.exitCode}</p>
                <p className="subtext">Dataset: {result.dataset_path}</p>
              </ResultCard>

              <ResultCard title="LLM Suggestion">
                <p className="big-text">{result.llm_result?.suggested_model || '—'}</p>
                <p className="subtext">Problem: {result.llm_result?.problem_type || '—'}</p>
                <p className="subtext">Reason: {result.llm_result?.reasoning || '—'}</p>
              </ResultCard>

              <ResultCard title="Script Resolution">
                <p className="subtext">Script path</p>
                <p className="mono">{result.script_path || 'Not found'}</p>
                {result.training && (
                  <p className="subtext">Training: {result.training.status} {result.training.model_path && `→ ${result.training.model_path}`}</p>
                )}
              </ResultCard>

              <ResultCard title="Dataset Stats">
                <pre className="stats-block">{result.stats || 'No stats returned.'}</pre>
              </ResultCard>
            </div>
          )}
        </section>
      </main>

      <section className="panel log-panel">
        <div className="panel-head">
          <h3>Raw logs</h3>
          <p>Direct stdout/stderr from <code>feature.py</code></p>
        </div>
        <pre className="log-block">{logs || 'No logs yet.'}</pre>
      </section>
    </div>
  );
}
