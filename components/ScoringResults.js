// src/components/ScoringResults.js
import React, { useState } from "react";

const API_BASE = process.env.REACT_APP_API_BASE_URL;

export default function ScoringResults() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleScore = async () => {
    setLoading(true);
    const res = await fetch(`${API_BASE}/score-cvs/`, { method: "POST" });
    const data = await res.json();
    setResults(data);
    setLoading(false);
  };

  return (
    <div className="scoring-section">
      <button onClick={handleScore} disabled={loading}>
        {loading ? "Scoring..." : "Score CVs"}
      </button>
      {results && (
        <div>
          <h3>Ranked Candidates</h3>
          <table>
            <thead>
              <tr>
                <th>CV ID</th>
                <th>Filename</th>
                <th>Score (%)</th>
                <th>Match Quality</th>
                <th>Max Similarity</th>
                <th>Avg Similarity</th>
                <th>Coverage</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody>
              {results.ranked_candidates.map((cv) => (
                <tr key={cv.cv_id}>
                  <td>{cv.cv_id}</td>
                  <td>{cv.filename}</td>
                  <td>{cv.final_score ?? "-"}</td>
                  <td>{cv.match_quality ?? "-"}</td>
                  <td>{cv.max_similarity ?? "-"}</td>
                  <td>{cv.avg_similarity ?? "-"}</td>
                  <td>{cv.coverage_score ?? "-"}</td>
                  <td>{cv.error ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4>Summary</h4>
          <pre>{JSON.stringify(results.summary, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
