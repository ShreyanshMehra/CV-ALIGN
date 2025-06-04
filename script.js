// DOM Elements
const jdDropZone = document.getElementById('jdDropZone');
const cvDropZone = document.getElementById('cvDropZone');
const jdInput = document.getElementById('jdInput');
const cvInput = document.getElementById('cvInput');
const jdFileInfo = document.getElementById('jdFileInfo');
const cvFileInfo = document.getElementById('cvFileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsContainer = document.getElementById('resultsContainer');
const loadingOverlay = document.getElementById('loadingOverlay');

// State
let jdFile = null;
let cvFiles = [];
const API_URL = 'http://127.0.0.1:8001/api';

// Utility Functions
function showLoading() {
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function updateLoadingMessage(message) {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.textContent = message;
    }
}

function updateAnalyzeButton() {
    analyzeBtn.disabled = !jdFile || cvFiles.length === 0;
}

async function promptForCVId(filename) {
    const id = prompt(`Please enter an ID for CV: ${filename}\nThis ID will be used to identify the CV in the analysis.`);
    if (!id) {
        throw new Error('CV ID is required');
    }
    return id.trim();
}

// Event Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.drop-zone').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.drop-zone').classList.remove('drag-over');
}

async function handleJDDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    jdDropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        jdFile = files[0];
        jdFileInfo.textContent = `Selected: ${jdFile.name}`;
        updateAnalyzeButton();
        // Upload JD immediately when selected
        await uploadJD(jdFile);
    }
}

async function handleCVDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    cvDropZone.classList.remove('drag-over');
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
        cvFiles = files;
        cvFileInfo.textContent = `Selected ${files.length} file(s): ${files.map(f => f.name).join(', ')}`;
        updateAnalyzeButton();
        // Upload CVs immediately when selected
        await uploadCVs(files);
    }
}

// File Input Change Handlers
jdInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        jdFile = e.target.files[0];
        jdFileInfo.textContent = `Selected: ${jdFile.name}`;
        updateAnalyzeButton();
        // Upload JD immediately when selected
        await uploadJD(jdFile);
    }
});

cvInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        cvFiles = Array.from(e.target.files);
        cvFileInfo.textContent = `Selected ${cvFiles.length} file(s): ${cvFiles.map(f => f.name).join(', ')}`;
        updateAnalyzeButton();
        // Upload CVs immediately when selected
        await uploadCVs(cvFiles);
    }
});

async function uploadJD(file) {
    try {
        showLoading();
        updateLoadingMessage('Uploading JD file...');
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/upload-jd`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to upload JD');
        }
        
        const result = await response.json();
        console.log('JD uploaded successfully:', result);
        jdFileInfo.textContent = `Uploaded: ${file.name}`;
    } catch (error) {
        console.error('Error uploading JD:', error);
        alert('Failed to upload JD. Please try again.');
        jdFileInfo.textContent = `Error uploading: ${file.name}`;
    } finally {
        hideLoading();
    }
}

async function uploadCVs(files) {
    try {
        showLoading();
        updateLoadingMessage(`Processing ${files.length} CV files...`);
        
        const uploadedFiles = [];
        const failedFiles = [];
        let currentFileNumber = 1;
        
        for (const file of files) {
            updateLoadingMessage(`Processing CV ${currentFileNumber} of ${files.length}: ${file.name}`);
            
            try {
                // Prompt for CV ID
                const cvId = await promptForCVId(file.name);
                
                updateLoadingMessage(`Uploading CV ${currentFileNumber} of ${files.length}: ${file.name}`);
                
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch(`${API_URL}/upload-cv?cv_id=${encodeURIComponent(cvId)}`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.detail || 'Upload failed');
                }
                
                console.log(`CV uploaded successfully:`, result);
                uploadedFiles.push({
                    name: file.name,
                    id: cvId
                });
            } catch (error) {
                console.error(`Error uploading CV ${file.name}:`, error);
                failedFiles.push({
                    name: file.name,
                    error: error.message
                });
            }
            
            currentFileNumber++;
        }
        
        // Update UI with results
        if (failedFiles.length > 0) {
            const failureMessage = failedFiles
                .map(f => `${f.name}: ${f.error}`)
                .join('\n');
            
            cvFileInfo.innerHTML = `
                <div class="upload-summary">
                    <div class="success">Successfully uploaded: ${uploadedFiles.length}</div>
                    <div class="failed">Failed: ${failedFiles.length}</div>
                    <div class="details">
                        ${uploadedFiles.map(f => `
                            <div class="success-item">✓ ${f.name}</div>
                        `).join('')}
                        ${failedFiles.map(f => `
                            <div class="error-item">✗ ${f.name}</div>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            cvFileInfo.innerHTML = `
                <div class="upload-summary success">
                    <div>Successfully uploaded all ${uploadedFiles.length} files:</div>
                    <div class="details">
                        ${uploadedFiles.map(f => `
                            <div class="success-item">✓ ${f.name}</div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        // Update the files array to only include successfully uploaded files
        cvFiles = uploadedFiles.map(f => ({name: f.name, id: f.id}));
        updateAnalyzeButton();
        
    } catch (error) {
        console.error('Error in CV upload process:', error);
        cvFileInfo.innerHTML = `<div class="error">Upload process failed: ${error.message}</div>`;
    } finally {
        hideLoading();
    }
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    try {
        showLoading();
        updateLoadingMessage('Analyzing resumes...');
        
        const response = await fetch(`${API_URL}/final-output`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Analysis failed: ${errorText}`);
        }
        
        const results = await response.json();
        
        // Transform the results to match the expected format
        const transformedResults = {
            rankings: {
                results: results.ranked_candidates.map(candidate => ({
                    cv_id: candidate.cv_id,
                    filename: candidate.filename,
                    relative_score: candidate.relative_score
                }))
            },
            detailed_analysis: {
                results: results.ranked_candidates.map(candidate => ({
                    cv_id: candidate.cv_id,
                    filename: candidate.filename,
                    review: candidate.pros_review?.review || [],
                    error: candidate.error
                }))
            }
        };
        
        displayResults(transformedResults);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
    } finally {
        hideLoading();
    }
});

function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    // Create tabs container
    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'tabs-container';
    
    // Create tab buttons
    const rankingTab = document.createElement('button');
    rankingTab.className = 'tab-button active';
    rankingTab.textContent = 'Rankings';
    
    const analysisTab = document.createElement('button');
    analysisTab.className = 'tab-button';
    analysisTab.textContent = 'Detailed Analysis';
    
    tabsContainer.appendChild(rankingTab);
    tabsContainer.appendChild(analysisTab);
    
    // Create content containers
    const rankingContent = document.createElement('div');
    rankingContent.className = 'tab-content active';
    
    const analysisContent = document.createElement('div');
    analysisContent.className = 'tab-content';
    
    // Populate Rankings tab
    if (results.rankings && results.rankings.results) {
        results.rankings.results.forEach((result, index) => {
            const rankCard = document.createElement('div');
            rankCard.className = 'result-card ranking-card';
            rankCard.innerHTML = `
                <div class="rank">#${index + 1}</div>
                <div class="cv-info">
                    <h3>CV ID: ${result.cv_id}</h3>
                    <div class="score">Relative Score: ${result.relative_score}%</div>
                    <div class="filename">File: ${result.filename}</div>
                </div>
            `;
            rankingContent.appendChild(rankCard);
        });
    }
    
    // Populate Analysis tab
    if (results.detailed_analysis && results.detailed_analysis.results) {
        results.detailed_analysis.results.forEach(result => {
            const analysisCard = document.createElement('div');
            analysisCard.className = 'result-card analysis-card';
            
            let reviewHtml = '';
            if (result.review && result.review.length > 0) {
                reviewHtml = `
                    <div class="review">
                        <ul>
                            ${result.review.map(point => `<li>${point}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            analysisCard.innerHTML = `
                <h3>CV ID: ${result.cv_id}</h3>
                <div class="filename">File: ${result.filename}</div>
                ${result.error ? `<div class="error">Error: ${result.error}</div>` : ''}
                ${reviewHtml}
            `;
            analysisContent.appendChild(analysisCard);
        });
    }
    
    // Add everything to the container
    resultsContainer.appendChild(tabsContainer);
    resultsContainer.appendChild(rankingContent);
    resultsContainer.appendChild(analysisContent);
    
    // Add tab switching functionality
    rankingTab.addEventListener('click', () => {
        rankingTab.classList.add('active');
        analysisTab.classList.remove('active');
        rankingContent.classList.add('active');
        analysisContent.classList.remove('active');
    });
    
    analysisTab.addEventListener('click', () => {
        analysisTab.classList.add('active');
        rankingTab.classList.remove('active');
        analysisContent.classList.add('active');
        rankingContent.classList.remove('active');
    });
}

// Setup drag and drop event listeners
[jdDropZone, cvDropZone].forEach(dropZone => {
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', dropZone === jdDropZone ? handleJDDrop : handleCVDrop);
});

// Setup click handlers for upload buttons
document.querySelectorAll('.upload-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const input = btn.nextElementSibling;
        input.click();
    });
}); 