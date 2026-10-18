import React, { useState } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie
} from 'recharts';
import { Activity, ShieldCheck, AlertCircle, Info, TrendingUp, Heart, ChevronRight } from 'lucide-react';

const FEATURE_LABELS = {
  age: "Age",
  sex: "Gender (1=M, 0=F)",
  cp: "Chest Pain Type",
  trestbps: "Resting BP",
  chol: "Cholesterol",
  fbs: "Fasting Blood Sugar",
  restecg: "Resting ECG",
  thalach: "Max Heart Rate",
  exang: "Exercise Angina",
  oldpeak: "ST Depression",
  slope: "Slope",
  ca: "Major Vessels",
  thal: "Thal"
};

const App = () => {
  const [formData, setFormData] = useState({
    age: 57, sex: 1, cp: 0, trestbps: 140, chol: 241, fbs: 0, 
    restecg: 1, thalach: 123, exang: 1, oldpeak: 0.2, slope: 1, ca: 0, thal: 3
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: parseFloat(value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // For local dev, using localhost:8000
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError("Failed to fetch prediction. Ensure backend is running.");
      console.error(err);
      // Mock result for demo if backend is not available
      /*
      setResult({
        prediction: 1,
        probability: 0.87,
        risk_level: "High Risk",
        consistency_score: 80,
        shap: {
          shap_values: [0.1, -0.05, 0.2, 0.05, 0.15, -0.02, 0.01, -0.1, 0.1, 0.05, 0.02, 0.12, 0.08],
          feature_names: Object.keys(FEATURE_LABELS)
        },
        lime: {
          exp_list: [["Chest Pain Type > 1", 0.25], ["Cholesterol > 240", 0.15], ["Max Heart Rate < 130", 0.1]]
        }
      });
      */
    } finally {
      setLoading(false);
    }
  };

  const shapData = result?.shap ? result.shap.feature_names.map((name, index) => ({
    name: FEATURE_LABELS[name] || name,
    value: result.shap.shap_values[index]
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : [];

  const limeData = result?.lime?.exp_list.map(item => ({
    name: item[0],
    value: item[1]
  })) || [];

  return (
    <div className="container">
      <header style={{ marginBottom: '40px', textAlign: 'center' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '10px', background: 'linear-gradient(to right, #6366f1, #a855f7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Explainable AI Heart Risk Dashboard
        </h1>
        <p style={{ color: 'var(--text-muted)' }}>Using XGBoost, SHAP, and LIME for Transparent Medical Predictions</p>
      </header>

      <div className="grid-cols-2" style={{ gridTemplateColumns: result ? '1fr 1.5fr' : '1fr' }}>
        {/* Input Form */}
        <div className="glass-card animate-fade-in" style={{ padding: '24px' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Activity color="#6366f1" /> Patient Metrics
          </h2>
          <form onSubmit={handleSubmit}>
            <div className="grid-cols-2">
              {Object.keys(formData).map(key => (
                <div key={key}>
                  <label>{FEATURE_LABELS[key] || key}</label>
                  <input 
                    type="number" 
                    step="any"
                    name={key} 
                    value={formData[key]} 
                    onChange={handleInputChange} 
                    className="input-field" 
                  />
                </div>
              ))}
            </div>
            <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '24px' }} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Risk"}
            </button>
          </form>
          {error && <p style={{ color: 'var(--danger)', marginTop: '10px', fontSize: '0.875rem' }}>{error}</p>}
        </div>

        {/* Results Section */}
        {result && (
          <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Risk Summary */}
            <div className="glass-card" style={{ padding: '24px', borderLeft: `6px solid ${result.prediction === 1 ? 'var(--danger)' : 'var(--success)'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textTransform: 'uppercase' }}>Risk Prediction</p>
                  <h3 style={{ fontSize: '2rem', color: result.prediction === 1 ? 'var(--danger)' : 'var(--success)' }}>
                    {result.risk_level} ({(result.probability * 100).toFixed(1)}%)
                  </h3>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Consistency Score</p>
                  <h3 style={{ color: 'var(--accent)' }}>{result.consistency_score.toFixed(0)}%</h3>
                </div>
              </div>
            </div>

            {/* SHAP Explanation */}
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                <TrendingUp color="#8b5cf6" /> SHAP Feature Contribution
              </h3>
              <div style={{ height: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ left: 20, right: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} fontSize={12} />
                    <Tooltip 
                      contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Bar dataKey="value">
                      {shapData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* LIME Explanation */}
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
                <ShieldCheck color="#10b981" /> LIME Local Explanation
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {limeData.map((item, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.05)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ flex: 1, fontSize: '0.9rem' }}>{item.name}</div>
                    <div style={{ fontWeight: 'bold', color: item.value > 0 ? 'var(--danger)' : 'var(--success)' }}>
                      {item.value > 0 ? '+' : ''}{(item.value * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <footer style={{ marginTop: '60px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
        <p>&copy; 2026 XAI Healthcare Framework. Built for Nit Goa.</p>
      </footer>
    </div>
  );
};

export default App;
