// src/components/FileUpload.js
import React, { useState } from "react";

const CLOUDINARY_URL = `https://api.cloudinary.com/v1_1/${process.env.REACT_APP_CLOUDINARY_CLOUD_NAME}/auto/upload`;
const UPLOAD_PRESET = process.env.REACT_APP_CLOUDINARY_UPLOAD_PRESET;
const API_BASE = process.env.REACT_APP_API_BASE_URL;

export default function FileUpload({ type, onUploaded }) {
  const [file, setFile] = useState(null);
  const [cloudUrl, setCloudUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  // Upload to Cloudinary for preview/storage
  const uploadToCloudinary = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("upload_preset", UPLOAD_PRESET);
    setUploading(true);
    setError("");
    try {
      const res = await fetch(CLOUDINARY_URL, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.secure_url) {
        setCloudUrl(data.secure_url);
        return data.secure_url;
      } else {
        throw new Error("Cloudinary upload failed");
      }
    } catch (e) {
      setError("Cloudinary upload failed");
      return "";
    } finally {
      setUploading(false);
    }
  };

  // Upload to backend for processing
  const uploadToBackend = async (file, cvId = "") => {
    const formData = new FormData();
    formData.append("file", file);
    let endpoint = "";
    if (type === "cv") {
      endpoint = `/upload-cv/?cv_id=${cvId}`;
    } else {
      endpoint = `/upload-jd/`;
    }
    setUploading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.detail) throw new Error(data.detail);
      onUploaded(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    setFile(file);
    if (file) {
      await uploadToCloudinary(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    let cvId = "";
    if (type === "cv") {
      cvId = prompt("Enter CV ID (unique for each candidate):");
      if (!cvId) return;
    }
    await uploadToBackend(file, cvId);
  };

  return (
    <div className="upload-section">
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".pdf,.docx" onChange={handleFileChange} />
        <button type="submit" disabled={uploading || !file}>
          {uploading ? "Uploading..." : `Upload ${type === "cv" ? "CV" : "JD"}`}
        </button>
      </form>
      {cloudUrl && (
        <div>
          <p>Preview (Cloudinary):</p>
          <a href={cloudUrl} target="_blank" rel="noopener noreferrer">
            {cloudUrl}
          </a>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
