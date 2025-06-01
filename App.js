import React from "react";
import FileUpload from "./components/FileUpload";
import Status from "./components/Status";
import ScoringResults from "./components/ScoringResults";
import "./App.css";

function App() {
  return (
    <div className="container">
      <h1>Enhanced CV Scoring System</h1>
      <div className="upload-grid">
        <FileUpload type="cv" onUploaded={() => window.location.reload()} />
        <FileUpload type="jd" onUploaded={() => window.location.reload()} />
      </div>
      <Status />
      <ScoringResults />
    </div>
  );
}

export default App;
