import { useEffect, useState } from "react";

function App() {
  const [top, setTop] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/top?n=10")
      .then(res => res.json())
      .then(data => {
        setTop(data);
        setLoading(false);
      })
      .catch(err => console.error(err));
  }, []);

  if (loading) return <h2>Loading...</h2>;

  return (
    <div style={{ padding: "20px" }}>
      <h1>Top 10 Subsidy Applicants</h1>
      <table border="1" cellPadding="5">
        <thead>
          <tr>
            <th>Applicant</th>
            <th>Region</th>
            <th>Final Score</th>
            <th>Efficiency</th>
            <th>Success Rate</th>
          </tr>
        </thead>
        <tbody>
          {top.map((row, idx) => (
            <tr key={idx}>
              <td>{row.akimat}</td>
              <td>{row.region}</td>
              <td>{row.final_score.toFixed(2)}</td>
              <td>{row.efficiency}</td>
              <td>{(row.success_rate * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
