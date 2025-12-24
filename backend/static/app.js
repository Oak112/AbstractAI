// Context Compiler - Frontend Application (streaming version)

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

// DOM Elements
const contextInput = document.getElementById('contextInput');
const charCount = document.getElementById('charCount');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const projectName = document.getElementById('projectName');
const generateBtn = document.getElementById('generateBtn');
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
	            modelToggleButtons.forEach((b) => b.classList.remove('active'));
	            btn.classList.add('active');
	            selectedModel = btn.dataset.model || 'gpt-5';
	        });
	    });
}

// File upload
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
        const text = await file.text();
        contextInput.value = text;
        charCount.textContent = text.length;
    }
});

// Generate documents (streaming via /api/generate-stream)
async function generateDocuments() {
	    const context = contextInput.value.trim();

		    if (!context) {
		        showToast('请输入一些上下文内容', 'error');
		        return;
		    }

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
	    heroInputCard.style.display = 'none';
	    loadingSection.style.display = 'none';
	    resultsSection.style.display = 'block';
	    documentsGrid.innerHTML = '';
	    generationStatusEl.textContent = '正在连接 AI 服务...';
	    generationProgressFill.style.width = '0%';
	    generationProgressText.textContent = '';
	    generateBtn.disabled = true;

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
	        heroInputCard.style.display = 'block';
	        resultsSection.style.display = 'none';
	        showToast(`生成失败: ${error.message || error}`, 'error');
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
	                renderStreamDocuments();
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
	            break;
	        }
	        case 'error': {
	            generationStatusEl.textContent = '生成失败, 请稍后重试';
	            showToast(event.message || '生成失败', 'error');
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

// Render streaming documents grid using current docStates
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
	                ? '\u5df2\u5b8c\u6210'
	                : doc.status === 'streaming'
	                ? '\u751f\u6210\u4e2d...'
	                : '\u7b49\u5f85\u751f\u6210';

	        const showActions = doc.status === 'done';

	        // Reuse existing card element if present to avoid full reflow/jitter during streaming
	        let card = documentsGrid.querySelector(`.document-card[data-doc-index="${index}"]`);
	        if (!card) {
	            card = document.createElement('div');
	            card.className = 'document-card';
	            card.dataset.docIndex = index.toString();
	            documentsGrid.appendChild(card);
	        }

	        // Attach / refresh click handler idempotently
	        card.onclick = () => openDocument(index);

	        card.innerHTML = `
	            <div class="doc-icon">
	                <i class="fas ${icon}"></i>
	            </div>
	            <h4>${doc.name}</h4>
	            <p class="doc-preview">${preview || '\u5185\u5bb9\u5c06\u5728\u8fd9\u91cc\u5b9e\u65f6\u663e\u793a...'}</p>
	            <p class="doc-size">${sizeKb ? sizeKb + ' KB' : '\u6682\u65e0\u5185\u5bb9'}</p>
	            <div class="doc-footer">
	                <div class="doc-status ${statusClass}">
	                    <span class="doc-status-dot"></span>
	                    <span>${statusLabel}</span>
	                </div>
	                ${
	                    showActions
	                        ? `<div class="doc-actions">
	                                <button class="doc-action-btn" title="\u590d\u5236" onclick="copyDocAtIndex(${index}, event)">
	                                    <i class="fas fa-copy"></i>
	                                </button>
	                                <button class="doc-action-btn" title="\u4e0b\u8f7d" onclick="downloadDocAtIndex(${index}, event)">
	                                    <i class="fas fa-download"></i>
	                                </button>
	                           </div>`
	                        : ''
	                }
	            </div>
	        `;
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

