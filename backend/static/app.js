// Abstraction AI - Frontend Application (streaming version)

// ============================================
// Analytics Helper Functions (GA4 + Clarity)
// ============================================

// Track custom events to GA4
function trackEvent(eventName, params = {}) {
    if (typeof gtag === 'function') {
        gtag('event', eventName, params);
    }
    // Also log to console in development
    console.debug('[Analytics]', eventName, params);
}

// Get or create anonymous user ID for tracking
function getAnonymousUserId() {
    let userId = localStorage.getItem('abstraction_user_id');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('abstraction_user_id', userId);
    }
    return userId;
}

// Track user properties
function getUserStats() {
    const stats = JSON.parse(localStorage.getItem('abstraction_stats') || '{}');
    return {
        total_generations: stats.total_generations || 0,
        total_downloads: stats.total_downloads || 0,
        first_visit: stats.first_visit || null,
        last_visit: stats.last_visit || null,
        nps_submitted: stats.nps_submitted || false,
        feedback_given: stats.feedback_given || false
    };
}

function updateUserStats(updates) {
    const stats = getUserStats();
    Object.assign(stats, updates);
    localStorage.setItem('abstraction_stats', JSON.stringify(stats));
}

// Initialize user tracking on page load
function initAnalytics() {
    const stats = getUserStats();
    const now = new Date().toISOString();

    if (!stats.first_visit) {
        updateUserStats({ first_visit: now });
        trackEvent('first_visit', { user_id: getAnonymousUserId() });
    }

    updateUserStats({ last_visit: now });

    // Track page view with user properties
    trackEvent('page_view_enhanced', {
        user_id: getAnonymousUserId(),
        is_returning: !!stats.first_visit,
        total_generations: stats.total_generations,
        referrer: document.referrer || 'direct'
    });
}

// Call on page load
initAnalytics();

// ============================================
// End Analytics Helper Functions
// ============================================

// Final documents (used for download, legacy compatibility)
let generatedDocuments = [];

// Currently opened document (object reference into docStates or generatedDocuments)
let currentDocument = null;
let currentDocumentIndex = null;

// Streaming state for 11 spec documents
let docStates = []; // [{ name, status: 'pending'|'streaming'|'done', content: '' }]
let totalDocsExpected = 0;
let docsCompleted = 0;
let streamDocumentNames = [];
let isStreaming = false;

// Model selection (gpt-5 / gemini-2.5-pro)
let selectedModel = 'gpt-5';

// Throttle rendering to reduce jitter (render at most every 100ms)
let renderPending = false;
let lastRenderTime = 0;
const RENDER_THROTTLE_MS = 100;

// Elapsed timer state
let elapsedTimerInterval = null;
let generationStartTime = null;

// DOM Elements
    const contextInput = document.getElementById('contextInput');
    const charCount = document.getElementById('charCount');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const projectName = document.getElementById('projectName');
    const generateBtn = document.getElementById('generateBtn');
    const heroContent = document.getElementById('heroContent');
    const heroInputCard = document.getElementById('heroInputCard');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const documentsGrid = document.getElementById('documentsGrid');
const modal = document.getElementById('documentModal');
const modalTitle = document.getElementById('modalTitle');
const modalContent = document.getElementById('modalContent');
const toast = document.getElementById('toast');
const generationStatusEl = document.getElementById('generationStatus');
const generationProgressFill = document.getElementById('generationProgressFill');
const generationProgressText = document.getElementById('generationProgressText');
const elapsedTimerEl = document.getElementById('elapsedTimer');
const modelToggleButtons = document.querySelectorAll('.model-toggle');

// Icon mapping for different document types
const DOCUMENT_ICONS = {
    '00': 'fa-eye',
    '01': 'fa-list-check',
    '02': 'fa-code',
    '03': 'fa-book',
    '04': 'fa-tasks',
    '05': 'fa-gavel',
    '06': 'fa-exclamation-triangle',
    '07': 'fa-quote-left',
    '08': 'fa-project-diagram',
    '09': 'fa-question-circle',
    '10': 'fa-not-equal'
};

// Character count
contextInput.addEventListener('input', () => {
	    charCount.textContent = contextInput.value.length;
});

// Model toggle
if (modelToggleButtons && modelToggleButtons.length) {
	    modelToggleButtons.forEach((btn) => {
	        btn.addEventListener('click', () => {
	            const previousModel = selectedModel;
	            modelToggleButtons.forEach((b) => b.classList.remove('active'));
	            btn.classList.add('active');
	            selectedModel = btn.dataset.model || 'gpt-5';

	            // Track model switch
	            if (previousModel !== selectedModel) {
	                trackEvent('model_switch', {
	                    from_model: previousModel,
	                    to_model: selectedModel
	                });
	            }
	        });
	    });
}

// File upload - support multiple files
let uploadedFiles = [];

fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    // Add new files to the list
    const fileTypes = [];
    for (const file of files) {
        // Avoid duplicates
        if (!uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            uploadedFiles.push(file);
            fileTypes.push(file.name.split('.').pop().toLowerCase());
        }
    }

    // Track file upload event
    trackEvent('file_upload', {
        file_count: files.length,
        total_files: uploadedFiles.length,
        file_types: fileTypes.join(','),
        total_size_kb: Math.round(uploadedFiles.reduce((sum, f) => sum + f.size, 0) / 1024)
    });

    // Update UI
    renderFileList();

    // Combine all file contents
    await combineFileContents();
});

function renderFileList() {
    if (uploadedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }

    fileList.innerHTML = uploadedFiles.map((file, index) => `
        <div class="file-item">
            <span class="file-item-name"><i class="fas fa-file-alt"></i> ${file.name}</span>
            <button type="button" class="file-remove-btn" data-index="${index}" title="移除">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');

    // Add remove handlers
    fileList.querySelectorAll('.file-remove-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const index = parseInt(btn.dataset.index);
            uploadedFiles.splice(index, 1);
            renderFileList();
            await combineFileContents();
        });
    });
}

async function combineFileContents() {
    if (uploadedFiles.length === 0) {
        contextInput.value = '';
        charCount.textContent = '0';
        return;
    }

    const contents = [];
    for (const file of uploadedFiles) {
        try {
            const text = await file.text();
            contents.push(`=== ${file.name} ===\n${text}`);
        } catch (err) {
            console.error(`Error reading ${file.name}:`, err);
            contents.push(`=== ${file.name} ===\n[无法读取文件内容]`);
        }
    }

    const combined = contents.join('\n\n');
    contextInput.value = combined;
    charCount.textContent = combined.length;
}

// Generate documents (streaming via /api/generate-stream)
async function generateDocuments() {
	    const context = contextInput.value.trim();

		    if (!context) {
		        showToast('请输入一些上下文内容', 'error');
		        trackEvent('generate_attempt_empty');
		        return;
		    }

	    // Track generation start
	    const inputLength = context.length;
	    trackEvent('generate_start', {
	        input_length: inputLength,
	        input_length_bucket: inputLength < 1000 ? 'short' : inputLength < 5000 ? 'medium' : 'long',
	        file_count: uploadedFiles.length,
	        model: selectedModel,
	        project_name_set: !!projectName.value
	    });

	    // Reset state
	    isStreaming = true;
	    generatedDocuments = [];
	    docStates = [];
	    streamDocumentNames = [];
	    totalDocsExpected = 0;
	    docsCompleted = 0;
	    currentDocument = null;
	    currentDocumentIndex = null;

	    // Prepare UI
	    if (heroContent) {
	        heroContent.style.display = 'none';
	    }
	    heroInputCard.style.display = 'none';
	    loadingSection.style.display = 'none';
	    resultsSection.style.display = 'block';
	    documentsGrid.innerHTML = '';
	    generationStatusEl.textContent = '正在连接 AI 服务...';
	    generationProgressFill.style.width = '0%';
	    generationProgressText.textContent = '';
	    generateBtn.disabled = true;

	    // Start elapsed timer
	    clearElapsedTimer();
	    startElapsedTimer();

	    try {
	        const response = await fetch('/api/generate-stream', {
	            method: 'POST',
	            headers: {
	                'Content-Type': 'application/json'
	            },
	            body: JSON.stringify({
	                context: context,
	                project_name: projectName.value || '未命名项目',
	                model: selectedModel,
	            })
	        });

	        if (!response.ok || !response.body) {
	            let errorDetail = '';
	            try {
	                const data = await response.json();
	                errorDetail = data.detail || '';
	            } catch (e) {}
	            throw new Error(errorDetail || '生成失败');
	        }

	        const reader = response.body.getReader();
	        const decoder = new TextDecoder('utf-8');
	        let buffer = '';

	        while (true) {
	            const { value, done } = await reader.read();
	            if (done) break;
	            buffer += decoder.decode(value, { stream: true });

	            let newlineIndex;
	            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
	                const line = buffer.slice(0, newlineIndex).trim();
	                buffer = buffer.slice(newlineIndex + 1);
	                if (!line) continue;
	                let event;
	                try {
	                    event = JSON.parse(line);
	                } catch (e) {
	                    console.error('Bad stream event line:', line);
	                    continue;
	                }
	                handleStreamEvent(event);
	            }
	        }

	    } catch (error) {
	        console.error('Generation error:', error);
	        if (heroContent) {
	            heroContent.style.display = 'block';
	        }
	        heroInputCard.style.display = 'block';
	        resultsSection.style.display = 'none';
	        showToast(`生成失败: ${error.message || error}`, 'error');
	        clearElapsedTimer();
	    } finally {
	        isStreaming = false;
	        generateBtn.disabled = false;
	    }
	}

// Handle streaming events from /api/generate-stream
function handleStreamEvent(event) {
	    switch (event.type) {
	        case 'meta': {
	            streamDocumentNames = event.document_names || [];
	            totalDocsExpected = streamDocumentNames.length;
	            docsCompleted = 0;
	            docStates = streamDocumentNames.map((name) => ({
	                name,
	                status: 'pending',
	                content: ''
	            }));
	            generationStatusEl.textContent = `已准备生成 ${totalDocsExpected} 份文档`;
	            updateGenerationProgress();
	            renderStreamDocuments();
	            break;
	        }
	        case 'doc_started': {
	            const idx = event.doc_index;
	            if (docStates[idx]) {
	                docStates[idx].status = 'streaming';
	                generationStatusEl.textContent = `正在生成: ${docStates[idx].name}`;
	                renderStreamDocuments();
	            }
	            break;
	        }
	        case 'chunk': {
	            const idx = event.doc_index;
	            const delta = event.delta || '';
	            if (docStates[idx]) {
	                docStates[idx].content += delta;
	                // If this doc is open in modal, append live
	                if (currentDocumentIndex === idx && modal.classList.contains('active')) {
	                    modalContent.textContent += delta;
	                    modalContent.scrollTop = modalContent.scrollHeight;
	                }
	                // Throttle card updates to reduce jitter
	                throttledRenderDocuments();
	            }
	            break;
	        }
	        case 'doc_complete': {
	            const idx = event.doc_index;
	            if (docStates[idx] && docStates[idx].status !== 'done') {
	                docStates[idx].status = 'done';
	                docsCompleted += 1;
	                updateGenerationProgress();
	                renderStreamDocuments();
	            }
	            break;
	        }
	        case 'done': {
	            // Finalize generatedDocuments for downloadAll / legacy APIs
	            generatedDocuments = docStates.map((d) => ({ name: d.name, content: d.content }));
	            generationStatusEl.textContent = `已生成完成 ${docsCompleted} / ${totalDocsExpected} 份文档`;
	            // Stop timer but keep final time displayed
	            stopElapsedTimer();

	            // Calculate generation time
	            const generationTime = generationStartTime ? Math.round((Date.now() - generationStartTime) / 1000) : 0;
	            const totalContentLength = docStates.reduce((sum, d) => sum + (d.content?.length || 0), 0);

	            // Track generation complete
	            trackEvent('generate_complete', {
	                document_count: docsCompleted,
	                generation_time_seconds: generationTime,
	                total_content_length: totalContentLength,
	                model: selectedModel
	            });

	            // Update user stats
	            const stats = getUserStats();
	            updateUserStats({ total_generations: stats.total_generations + 1 });

	            // Show feedback widget
	            showFeedbackWidget();

	            // Maybe show NPS (after 3+ generations, if not already submitted)
	            maybeShowNPS();
	            break;
	        }
	        case 'error': {
	            generationStatusEl.textContent = '生成失败, 请稍后重试';
	            showToast(event.message || '生成失败', 'error');
	            clearElapsedTimer();

	            // Track error
	            trackEvent('generate_error', {
	                error_message: event.message || 'unknown',
	                model: selectedModel
	            });
	            break;
	        }
	        case 'heartbeat': {
	            // Keep-alive event from Gemini non-streaming mode
	            const msg = event.message || `等待中... (${event.elapsed_seconds || 0}秒)`;
	            generationStatusEl.textContent = msg;
	            console.debug('Heartbeat:', event);
	            break;
	        }
	        default:
	            console.debug('Unknown stream event', event);
	    }
}

function updateGenerationProgress() {
	    if (!totalDocsExpected) {
	        generationProgressFill.style.width = '0%';
	        generationProgressText.textContent = '';
	        return;
	    }
	    const percent = Math.round((docsCompleted / totalDocsExpected) * 100);
	    generationProgressFill.style.width = `${percent}%`;
	    generationProgressText.textContent = `${docsCompleted} / ${totalDocsExpected}`;
}

// Throttled version for chunk updates
function throttledRenderDocuments() {
    const now = Date.now();
    if (now - lastRenderTime < RENDER_THROTTLE_MS) {
        if (!renderPending) {
            renderPending = true;
            setTimeout(() => {
                renderPending = false;
                lastRenderTime = Date.now();
                renderStreamDocuments();
            }, RENDER_THROTTLE_MS - (now - lastRenderTime));
        }
        return;
    }
    lastRenderTime = now;
    renderStreamDocuments();
}

// Render streaming documents grid using current docStates
// Optimized: create card structure once, then only update changed parts
function renderStreamDocuments() {
    if (!docStates.length) return;

    docStates.forEach((doc, index) => {
        const prefix = doc.name.substring(0, 2);
        const icon = DOCUMENT_ICONS[prefix] || 'fa-file-alt';
        const previewSource = doc.content || '';
        const preview = previewSource.substring(0, 150).replace(/[#\n]/g, ' ').trim();
        const sizeKb = previewSource ? (previewSource.length / 1024).toFixed(1) : null;

        const statusClass =
            doc.status === 'done'
                ? 'doc-status-done'
                : doc.status === 'streaming'
                ? 'doc-status-streaming'
                : 'doc-status-pending';

        const statusLabel =
            doc.status === 'done'
                ? '已完成'
                : doc.status === 'streaming'
                ? '生成中...'
                : '等待生成';

        const showActions = doc.status === 'done';

        // Reuse existing card element if present
        let card = documentsGrid.querySelector(`.document-card[data-doc-index="${index}"]`);
        const isNew = !card;

        if (isNew) {
            // Create card structure once
            card = document.createElement('div');
            card.className = 'document-card';
            card.dataset.docIndex = index.toString();
            card.innerHTML = `
                <div class="doc-icon">
                    <i class="fas ${icon}"></i>
                </div>
                <h4 class="doc-title"></h4>
                <p class="doc-preview"></p>
                <p class="doc-size"></p>
                <div class="doc-footer">
                    <div class="doc-status">
                        <span class="doc-status-dot"></span>
                        <span class="doc-status-label"></span>
                    </div>
                    <div class="doc-actions"></div>
                </div>
            `;
            // Single click handler attached once
            card.addEventListener('click', (e) => {
                // Don't open if clicking action buttons
                if (e.target.closest('.doc-action-btn')) return;
                openDocument(index);
            });
            documentsGrid.appendChild(card);
        }

        // Update only changed parts (no innerHTML replacement after initial)
        const titleEl = card.querySelector('.doc-title');
        const previewEl = card.querySelector('.doc-preview');
        const sizeEl = card.querySelector('.doc-size');
        const statusEl = card.querySelector('.doc-status');
        const statusLabelEl = card.querySelector('.doc-status-label');
        const actionsEl = card.querySelector('.doc-actions');

        if (titleEl && titleEl.textContent !== doc.name) {
            titleEl.textContent = doc.name;
        }

        const displayPreview = preview || '内容将在这里实时显示...';
        if (previewEl && previewEl.textContent !== displayPreview) {
            previewEl.textContent = displayPreview;
        }

        const displaySize = sizeKb ? sizeKb + ' KB' : '暂无内容';
        if (sizeEl && sizeEl.textContent !== displaySize) {
            sizeEl.textContent = displaySize;
        }

        if (statusEl) {
            statusEl.className = 'doc-status ' + statusClass;
        }
        if (statusLabelEl && statusLabelEl.textContent !== statusLabel) {
            statusLabelEl.textContent = statusLabel;
        }

        // Only update actions when status changes to done
        if (actionsEl) {
            const hasButtons = actionsEl.children.length > 0;
            if (showActions && !hasButtons) {
                actionsEl.innerHTML = `
                    <button class="doc-action-btn" title="复制" data-action="copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="doc-action-btn" title="下载" data-action="download">
                        <i class="fas fa-download"></i>
                    </button>
                `;
                // Add event listeners for action buttons
                const copyBtn = actionsEl.querySelector('[data-action="copy"]');
                const downloadBtn = actionsEl.querySelector('[data-action="download"]');
                if (copyBtn) {
                    copyBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        copyDocAtIndex(index, e);
                    });
                }
                if (downloadBtn) {
                    downloadBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        downloadDocAtIndex(index, e);
                    });
                }
            }
        }
    });
}

// Legacy non-stream display (for /api/generate fallback)
function displayDocuments(documents) {
	    docStates = documents.map((doc) => ({
	        name: doc.name,
	        status: 'done',
	        content: doc.content
	    }));
	    totalDocsExpected = docStates.length;
	    docsCompleted = totalDocsExpected;
	    updateGenerationProgress();
	    renderStreamDocuments();
	}

// Open document in modal
function openDocument(index) {
	    const source = docStates.length ? docStates : generatedDocuments;
	    const doc = source[index];
	    if (!doc) return;

	    currentDocumentIndex = index;
	    currentDocument = doc;
	    const titleSuffix = doc.status === 'streaming' ? '（生成中…）' : '';
	    modalTitle.textContent = (doc.name || '文档') + titleSuffix;
	    modalContent.textContent = doc.content || '文档内容正在生成中…';
	    modal.classList.add('active');
	    document.body.style.overflow = 'hidden';

	    // Track document view
	    trackEvent('document_view', {
	        document_name: doc.name,
	        document_index: index,
	        content_length: doc.content?.length || 0
	    });
}

// Close modal
function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
    currentDocument = null;
}

// Copy document
function copyDocument() {
    if (currentDocument) {
        navigator.clipboard.writeText(currentDocument.content);
        showToast('已复制到剪贴板', 'success');

        // Track copy event
        trackEvent('document_copy', {
            document_name: currentDocument.name,
            content_length: currentDocument.content?.length || 0
        });
    }
}

// Download single document
function downloadDocument() {
    if (currentDocument) {
        const blob = new Blob([currentDocument.content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = currentDocument.name;
        a.click();
        URL.revokeObjectURL(url);
        showToast('下载已开始', 'success');

        // Track download event
        trackEvent('document_download', {
            document_name: currentDocument.name,
            content_length: currentDocument.content?.length || 0
        });

        // Update stats
        const stats = getUserStats();
        updateUserStats({ total_downloads: stats.total_downloads + 1 });
    }
}

// Per-card copy action (from results grid)
function copyDocAtIndex(index, event) {
    if (event) {
        event.stopPropagation();
    }
    const doc = docStates[index];
    if (!doc || doc.status !== 'done' || !doc.content) return;
    navigator.clipboard.writeText(doc.content);
    showToast('已复制到剪贴板', 'success');
}

// Per-card download action (from results grid)
function downloadDocAtIndex(index, event) {
    if (event) {
        event.stopPropagation();
    }
    const doc = docStates[index];
    if (!doc || doc.status !== 'done' || !doc.content) return;

    const blob = new Blob([doc.content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = doc.name;
    a.click();
    URL.revokeObjectURL(url);
    showToast('下载已开始', 'success');
}

// Download all documents as ZIP
async function downloadAll() {
	    if (!generatedDocuments.length && docStates.length) {
	        // If streaming has populated docStates but we haven't copied them yet
	        generatedDocuments = docStates.map((d) => ({ name: d.name, content: d.content }));
	    }
	    if (generatedDocuments.length === 0) return;

    // Track download all event
    const totalSize = generatedDocuments.reduce((sum, d) => sum + (d.content?.length || 0), 0);
    trackEvent('download_all', {
        document_count: generatedDocuments.length,
        total_size_kb: Math.round(totalSize / 1024)
    });

    // Update stats
    const stats = getUserStats();
    updateUserStats({ total_downloads: stats.total_downloads + generatedDocuments.length });

    try {
        const response = await fetch('/api/download-zip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(generatedDocuments)
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'context_compiler_output.zip';
            a.click();
            URL.revokeObjectURL(url);
            showToast('ZIP 下载已开始', 'success');
        } else {
            // Fallback: download files individually
            generatedDocuments.forEach(doc => {
                const blob = new Blob([doc.content], { type: 'text/plain;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = doc.name;
                a.click();
                URL.revokeObjectURL(url);
            });
            showToast('文件下载已开始', 'success');
        }
    } catch (error) {
        console.error('Download error:', error);
        showToast('下载失败', 'error');
    }
}

// Reset form
function resetForm() {
		    if (heroContent) {
		        heroContent.style.display = 'block';
		    }
		    heroInputCard.style.display = 'block';
	    resultsSection.style.display = 'none';
	    loadingSection.style.display = 'none';
	    generatedDocuments = [];
	    docStates = [];
	    streamDocumentNames = [];
	    totalDocsExpected = 0;
	    docsCompleted = 0;
	    currentDocument = null;
	    currentDocumentIndex = null;
	    generationStatusEl.textContent = '准备生成 11 份文档';
	    generationProgressFill.style.width = '0%';
	    generationProgressText.textContent = '';
	    clearElapsedTimer();
}

// Show toast notification
function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = 'toast show ' + type;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
    }
});

// Close modal on backdrop click
modal.addEventListener('click', (e) => {
    if (e.target === modal) {
        closeModal();
    }
});

// ========== Elapsed Timer Functions ==========

function startElapsedTimer() {
    generationStartTime = Date.now();
    updateElapsedTimer();
    elapsedTimerInterval = setInterval(updateElapsedTimer, 1000);
}

function stopElapsedTimer() {
    if (elapsedTimerInterval) {
        clearInterval(elapsedTimerInterval);
        elapsedTimerInterval = null;
    }
}

function updateElapsedTimer() {
    if (!generationStartTime || !elapsedTimerEl) return;
    const elapsed = Math.floor((Date.now() - generationStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    if (minutes > 0) {
        elapsedTimerEl.textContent = `已用时 ${minutes} 分 ${seconds} 秒`;
    } else {
        elapsedTimerEl.textContent = `已用时 ${seconds} 秒`;
    }
}

function clearElapsedTimer() {
    stopElapsedTimer();
    generationStartTime = null;
    if (elapsedTimerEl) {
        elapsedTimerEl.textContent = '';
    }
}

// ========== Copy AI Prompt ==========

function copyAiPrompt() {
    const promptTextarea = document.getElementById('aiPromptText');
    if (!promptTextarea) return;

    const text = promptTextarea.value;

    // More robust copy approach
    function fallbackCopy() {
        // Create a temporary textarea for copying
        const tempTextarea = document.createElement('textarea');
        tempTextarea.value = text;
        tempTextarea.style.position = 'fixed';
        tempTextarea.style.left = '-9999px';
        tempTextarea.style.top = '0';
        document.body.appendChild(tempTextarea);
        tempTextarea.focus();
        tempTextarea.select();

        try {
            document.execCommand('copy');
            showToast('Prompt 已复制到剪贴板', 'success');
        } catch (err) {
            showToast('复制失败，请手动复制', 'error');
        }

        document.body.removeChild(tempTextarea);
    }

    // Try modern clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Prompt 已复制到剪贴板', 'success');
        }).catch(() => {
            fallbackCopy();
        });
    } else {
        fallbackCopy();
    }
}

// ============================================
// Feedback Widget Functions
// ============================================

function showFeedbackWidget() {
    const stats = getUserStats();
    // Don't show if already given feedback this session
    if (stats.feedback_given) return;

    const widget = document.getElementById('feedbackWidget');
    if (widget) {
        setTimeout(() => {
            widget.style.display = 'flex';
        }, 2000); // Show after 2 seconds
    }
}

function submitFeedback(type) {
    trackEvent('feedback_submitted', {
        feedback_type: type,
        total_generations: getUserStats().total_generations
    });

    updateUserStats({ feedback_given: true });

    const widget = document.getElementById('feedbackWidget');
    if (widget) {
        widget.innerHTML = '<div class="feedback-thanks">感谢你的反馈！ 🎉</div>';
        setTimeout(() => {
            widget.style.display = 'none';
        }, 2000);
    }

    showToast(type === 'positive' ? '感谢你的支持！' : '感谢反馈，我们会继续改进', 'success');
}

function dismissFeedback() {
    const widget = document.getElementById('feedbackWidget');
    if (widget) {
        widget.style.display = 'none';
    }
    trackEvent('feedback_dismissed');
}

// ============================================
// NPS Survey Functions
// ============================================

let selectedNPSScore = null;

function maybeShowNPS() {
    const stats = getUserStats();

    // Show NPS after 3+ generations, if not already submitted
    if (stats.total_generations >= 3 && !stats.nps_submitted) {
        setTimeout(() => {
            showNPS();
        }, 5000); // Show after 5 seconds
    }
}

function showNPS() {
    const modal = document.getElementById('npsModal');
    if (modal) {
        modal.style.display = 'flex';
        trackEvent('nps_shown', {
            total_generations: getUserStats().total_generations
        });

        // Add click handlers to score buttons
        modal.querySelectorAll('.nps-score').forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove previous selection
                modal.querySelectorAll('.nps-score').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                selectedNPSScore = parseInt(btn.dataset.score);

                // Show feedback input
                document.getElementById('npsFeedbackInput').style.display = 'block';

                trackEvent('nps_score_selected', {
                    score: selectedNPSScore,
                    category: selectedNPSScore <= 6 ? 'detractor' : selectedNPSScore <= 8 ? 'passive' : 'promoter'
                });
            });
        });
    }
}

function closeNPS() {
    const modal = document.getElementById('npsModal');
    if (modal) {
        modal.style.display = 'none';
    }

    if (selectedNPSScore === null) {
        trackEvent('nps_dismissed');
    }
}

function submitNPSFeedback() {
    const feedbackText = document.getElementById('npsFeedbackText')?.value || '';

    trackEvent('nps_submitted', {
        score: selectedNPSScore,
        category: selectedNPSScore <= 6 ? 'detractor' : selectedNPSScore <= 8 ? 'passive' : 'promoter',
        has_feedback: feedbackText.length > 0,
        feedback_length: feedbackText.length
    });

    updateUserStats({ nps_submitted: true });

    closeNPS();
    showToast('感谢你的反馈！', 'success');
}

