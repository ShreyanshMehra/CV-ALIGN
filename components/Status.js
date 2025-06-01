// src/components/Status.js
import React, { useEffect, useState } from "react";

const API_BASE = process.env.REACT_APP_API_BASE_URL;

export default function Status() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then((res) => res.json())
      .then(setStatus);
  }, []);

  if (!status) return <div>Loading status...</div>;

  return (
    <div className="status-section">
      <h3>System Status</h3>
      <p>CVs Uploaded: {status.cvs_uploaded}</p>
      <p>JD Uploaded: {status.jd_uploaded ? "Yes" : "No"}</p>
      <ul>
        {status.cv_ids.map((id) => (
          <li key={id}>{id}</li>
        ))}
      </ul>
    </div>
  );
}
