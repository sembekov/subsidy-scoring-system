import { useEffect, useState } from "react";

function App() {
  const [top, setTop] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedApplicant, setSelectedApplicant] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [batchResults, setBatchResults] = useState(null);
  const [error, setError] = useState(null);

  // Загрузка топ-10 кандидатов
  useEffect(() => {
    fetchTopApplicants();
  }, []);

  const fetchTopApplicants = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/top?n=10");
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();
      setTop(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Запрос объяснения для конкретного заявителя
  const fetchExplanation = async (applicant) => {
    setSelectedApplicant(applicant);
    setShowExplanation(true);
    setExplanation(null);
    
    try {
      // Формируем запрос на основе данных заявителя
      const requestData = {
        applicant_data: {
          efficiency: applicant.efficiency || 1.0,
          success_rate: applicant.success_rate || 0.5,
          application_count: applicant.application_count || 1,
          total_subsidy: applicant.total_subsidy || 0,
          stability: applicant.stability || 0.5,
          eff_vs_region: applicant.eff_vs_region || 0
        }
      };
      
      const res = await fetch("http://127.0.0.1:8000/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData)
      });
      
      if (!res.ok) throw new Error("Failed to get explanation");
      const data = await res.json();
      setExplanation(data);
    } catch (err) {
      console.error(err);
      setExplanation({ error: err.message });
    }
  };

  // Batch scoring demo
  const runBatchScoring = async () => {
    setBatchResults(null);
    try {
      // Берем топ-5 для демо
      const applicants = top.slice(0, 5).map(a => ({
        id: a.akimat,
        efficiency: a.efficiency || 1.0,
        success_rate: a.success_rate || 0.5,
        application_count: a.application_count || 1,
        total_subsidy: a.total_subsidy || 0
      }));
      
      const res = await fetch("http://127.0.0.1:8000/score_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ applicants })
      });
      
      if (!res.ok) throw new Error("Batch scoring failed");
      const data = await res.json();
      setBatchResults(data);
    } catch (err) {
      console.error(err);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 75) return "#4caf50";
    if (score >= 50) return "#ff9800";
    return "#f44336";
  };

  const getRecommendationBadge = (rec) => {
    const colors = {
      "одобрить": "#4caf50",
      "требует проверки": "#ff9800",
      "отклонить": "#f44336"
    };
    return { backgroundColor: colors[rec] || "#999", color: "white", padding: "3px 10px", borderRadius: "12px", fontSize: "12px" };
  };

  if (loading) return (
    <div style={{ textAlign: "center", padding: "50px" }}>
      <h2>Loading subsidy scoring system...</h2>
      <p>🤖 AI is analyzing agricultural producers</p>
    </div>
  );

  if (error) return (
    <div style={{ textAlign: "center", padding: "50px", color: "red" }}>
      <h2>Error: {error}</h2>
      <button onClick={fetchTopApplicants}>Retry</button>
    </div>
  );

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif", maxWidth: "1400px", margin: "0 auto" }}>
      <div style={{ background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", color: "white", padding: "20px", borderRadius: "10px", marginBottom: "20px" }}>
        <h1>🌾 Subsidy Scoring System - AI for Government</h1>
        <p>Merit-based distribution of agricultural subsidies with explainable AI</p>
        <div style={{ display: "flex", gap: "10px", marginTop: "15px" }}>
          <span style={{ background: "rgba(255,255,255,0.2)", padding: "5px 10px", borderRadius: "5px" }}>✅ Explainable AI</span>
          <span style={{ background: "rgba(255,255,255,0.2)", padding: "5px 10px", borderRadius: "5px" }}>👥 Human-in-the-loop</span>
          <span style={{ background: "rgba(255,255,255,0.2)", padding: "5px 10px", borderRadius: "5px" }}>📊 Merit-based scoring</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        {/* Левая колонка - таблица */}
        <div style={{ background: "white", padding: "20px", borderRadius: "10px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "15px" }}>
            <h2>🏆 Top 10 Applicants</h2>
            <button onClick={runBatchScoring} style={{ padding: "8px 15px", background: "#667eea", color: "white", border: "none", borderRadius: "5px", cursor: "pointer" }}>
              Run Batch Scoring
            </button>
          </div>
          
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#f5f5f5", borderBottom: "2px solid #ddd" }}>
                  <th style={{ padding: "10px", textAlign: "left" }}>Applicant</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Region</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Score</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Efficiency</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Success Rate</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {top.map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid #eee" }}>
                    <td style={{ padding: "10px" }}><strong>{row.akimat}</strong></td>
                    <td style={{ padding: "10px" }}>{row.region}</td>
                    <td style={{ padding: "10px" }}>
                      <span style={{ 
                        background: getScoreColor(row.final_score), 
                        color: "white", 
                        padding: "3px 8px", 
                        borderRadius: "5px",
                        fontWeight: "bold"
                      }}>
                        {row.final_score.toFixed(1)}
                      </span>
                    </td>
                    <td style={{ padding: "10px" }}>{row.efficiency?.toFixed(2)}</td>
                    <td style={{ padding: "10px" }}>{(row.success_rate * 100).toFixed(1)}%</td>
                    <td style={{ padding: "10px" }}>
                      <button onClick={() => fetchExplanation(row)} style={{ 
                        padding: "5px 10px", 
                        background: "#2196f3", 
                        color: "white", 
                        border: "none", 
                        borderRadius: "3px",
                        cursor: "pointer",
                        fontSize: "12px"
                      }}>
                        Explain
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Правая колонка - объяснение */}
        <div style={{ background: "white", padding: "20px", borderRadius: "10px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)" }}>
          <h2>🔍 Explainable AI Decision</h2>
          
          {!showExplanation ? (
            <div style={{ textAlign: "center", padding: "50px", color: "#999" }}>
              <p>Click "Explain" on any applicant to see why the AI made this decision</p>
              <p style={{ fontSize: "14px", marginTop: "10px" }}>⚡ Explainability is required for GovTech AI systems</p>
            </div>
          ) : (
            <div>
              {selectedApplicant && (
                <div style={{ marginBottom: "20px", padding: "15px", background: "#f5f5f5", borderRadius: "8px" }}>
                  <h3>{selectedApplicant.akimat}</h3>
                  <p>Region: {selectedApplicant.region}</p>
                </div>
              )}
              
              {!explanation ? (
                <div style={{ textAlign: "center", padding: "30px" }}>
                  <div className="spinner">🤖 Analyzing...</div>
                  <p>Generating explainable AI decision...</p>
                </div>
              ) : explanation.error ? (
                <div style={{ color: "red", padding: "20px", textAlign: "center" }}>
                  <p>❌ Error: {explanation.error}</p>
                  <p>Make sure the backend API is running on port 8000</p>
                </div>
              ) : (
                <div>
                  {/* Score */}
                  <div style={{ textAlign: "center", marginBottom: "20px" }}>
                    <div style={{ fontSize: "48px", fontWeight: "bold", color: getScoreColor(explanation.final_score) }}>
                      {explanation.final_score?.toFixed(1)}
                    </div>
                    <div style={{ marginTop: "10px" }}>
                      <span style={getRecommendationBadge(explanation.recommendation)}>
                        {explanation.recommendation}
                      </span>
                      {explanation.human_review_needed && (
                        <span style={{ marginLeft: "10px", background: "#ff9800", color: "white", padding: "3px 10px", borderRadius: "12px", fontSize: "12px" }}>
                          👤 Human review required
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Summary */}
                  <div style={{ background: "#e3f2fd", padding: "15px", borderRadius: "8px", marginBottom: "20px" }}>
                    <strong>📋 AI Summary:</strong>
                    <p style={{ marginTop: "10px", lineHeight: "1.6" }}>{explanation.summary}</p>
                  </div>

                  {/* Positive factors */}
                  {explanation.positive_factors?.length > 0 && (
                    <div style={{ marginBottom: "20px" }}>
                      <h4 style={{ color: "#4caf50" }}>✅ Positive Factors</h4>
                      <ul style={{ listStyle: "none", padding: 0 }}>
                        {explanation.positive_factors.map((factor, idx) => (
                          <li key={idx} style={{ padding: "8px", background: "#e8f5e9", marginBottom: "5px", borderRadius: "5px" }}>
                            <strong>{factor.factor}</strong>
                            <span style={{ float: "right", color: "#4caf50" }}>+{factor.importance?.toFixed(2)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Negative factors */}
                  {explanation.negative_factors?.length > 0 && (
                    <div style={{ marginBottom: "20px" }}>
                      <h4 style={{ color: "#f44336" }}>⚠️ Negative Factors</h4>
                      <ul style={{ listStyle: "none", padding: 0 }}>
                        {explanation.negative_factors.map((factor, idx) => (
                          <li key={idx} style={{ padding: "8px", background: "#ffebee", marginBottom: "5px", borderRadius: "5px" }}>
                            <strong>{factor.factor}</strong>
                            <span style={{ float: "right", color: "#f44336" }}>{factor.importance?.toFixed(2)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Human decision button */}
                  <div style={{ marginTop: "20px", textAlign: "center", padding: "15px", borderTop: "1px solid #eee" }}>
                    <p style={{ fontSize: "12px", color: "#666", marginBottom: "10px" }}>
                      AI is not the sole source of truth. Final decision rests with the commission.
                    </p>
                    <button style={{ 
                      padding: "10px 20px", 
                      background: "#ff9800", 
                      color: "white", 
                      border: "none", 
                      borderRadius: "5px",
                      cursor: "pointer",
                      fontWeight: "bold"
                    }}>
                      👥 Send for Human Review
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Batch results modal */}
      {batchResults && (
        <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }}>
          <div style={{ background: "white", padding: "20px", borderRadius: "10px", maxWidth: "600px", maxHeight: "80%", overflow: "auto" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "15px" }}>
              <h2>📊 Batch Scoring Results</h2>
              <button onClick={() => setBatchResults(null)} style={{ fontSize: "20px", cursor: "pointer" }}>✕</button>
            </div>
            
            <h3>Shortlist (Top {batchResults.shortlist?.length})</h3>
            {batchResults.shortlist?.map((item, idx) => (
              <div key={idx} style={{ padding: "10px", marginBottom: "10px", background: "#f5f5f5", borderRadius: "5px" }}>
                <strong>{idx+1}. {item.applicant_id}</strong>
                <span style={{ float: "right", fontWeight: "bold", color: getScoreColor(item.final_score) }}>
                  Score: {item.final_score?.toFixed(1)}
                </span>
                <p style={{ fontSize: "12px", marginTop: "5px", color: "#666" }}>{item.summary}</p>
                <span style={getRecommendationBadge(item.recommendation)}>{item.recommendation}</span>
              </div>
            ))}
            
            <p style={{ marginTop: "15px", fontSize: "12px", color: "#666", textAlign: "center" }}>
              Total processed: {batchResults.total_processed} applicants
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
