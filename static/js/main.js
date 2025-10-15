document.addEventListener('DOMContentLoaded', () => {

    const PRELOADED_FONT_BASE64 = ""; 
    
    const App = {
        state: {
            currentStep: 1, files: [], websites: [], results: [],
            fontData: null, isProcessing: false, isMockMode: false,
            activeDetailNav: null
        },
        elements: {
            steps: [document.getElementById('step1'), document.getElementById('step2'), document.getElementById('step3'), document.getElementById('step4')],
            stepWrappers: [document.getElementById('step-wrapper-1'), document.getElementById('step-wrapper-2'), document.getElementById('step-wrapper-3'), document.getElementById('step-wrapper-4')],
            stepLines: [document.getElementById('step-line-1'), document.getElementById('step-line-2'), document.getElementById('step-line-3')],
            contents: [document.getElementById('step1-content'), document.getElementById('step2-content'), document.getElementById('step3-content'), document.getElementById('step4-content')],
            nextStep1Btn: document.getElementById('next-step1'), nextStep2Btn: document.getElementById('next-step2'),
            prevStep2Btn: document.getElementById('prev-step2'), dropArea: document.getElementById('drop-area'),
            fileInput: document.getElementById('file-input'), fileList: document.getElementById('file-list'),
            websiteList: document.getElementById('website-list'), addWebsiteBtn: document.getElementById('add-website-btn'),
            progressText: document.getElementById('progress-text'), resultsContainer: document.getElementById('results-container'),
            backToStartBtn: document.getElementById('back-to-start'), fontUploadBtn: document.getElementById('font-upload-btn'),
            fontInput: document.getElementById('font-input'), fontStatus: document.getElementById('font-status'),
            startPreviewBtn: document.getElementById('start-preview-btn'), 
            exitPreviewBtn: document.getElementById('exit-preview-btn'),
            previewBanner: document.getElementById('preview-banner'), 
            testConnectionBtn: document.getElementById('test-connection-btn'),
            connectionStatus: document.getElementById('connection-status'),
        },

        init() {
            this.ui.updateStepUI();
            this.events.bind();
            this.helpers.preloadDefaultFont();
        },
        
        events: {
            bind() {
                App.elements.nextStep1Btn.addEventListener('click', () => App.ui.changeStep(2));
                App.elements.prevStep2Btn.addEventListener('click', () => App.ui.changeStep(1));
                App.elements.nextStep2Btn.addEventListener('click', () => App.logic.startProcessing());
                App.elements.backToStartBtn.addEventListener('click', () => App.logic.resetApp());
                App.elements.dropArea.addEventListener('click', () => App.elements.fileInput.click());
                App.elements.fileInput.addEventListener('change', (e) => App.logic.handleFiles(e.target.files));
                const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    App.elements.dropArea.addEventListener(eventName, preventDefaults, false);
                });
                ['dragenter', 'dragover'].forEach(eventName => {
                    App.elements.dropArea.addEventListener(eventName, () => App.elements.dropArea.classList.add('dragover'), false);
                });
                ['dragleave', 'drop'].forEach(eventName => {
                    App.elements.dropArea.addEventListener(eventName, () => App.elements.dropArea.classList.remove('dragover'), false);
                });
                App.elements.dropArea.addEventListener('drop', (e) => App.logic.handleFiles(e.dataTransfer.files), false);
                App.elements.addWebsiteBtn.addEventListener('click', () => App.logic.addWebsite('', ''));
                App.elements.fontUploadBtn.addEventListener('click', () => App.elements.fontInput.click());
                App.elements.fontInput.addEventListener('change', (e) => App.logic.handleFontFile(e.target.files[0]));
                App.elements.startPreviewBtn.addEventListener('click', () => App.logic.startPreviewMode());
                App.elements.exitPreviewBtn.addEventListener('click', () => App.logic.exitPreviewMode());
                App.elements.testConnectionBtn.addEventListener('click', () => App.logic.testBackendConnection());
            }
        },

        logic: {
            handleFiles(files) {
                if (App.state.isMockMode) { App.state.files = []; App.state.websites = []; }
                for (const file of files) {
                    if (file.type === 'application/pdf') {
                        const fileObject = { file: file, id: 'file-' + Date.now() + Math.random(), name: file.name, size: file.size };
                        App.state.files.push(fileObject);
                        const companyName = file.name.replace(/\.pdf$/i, '');
                        if (!App.state.websites.some(w => w.company === companyName)) {
                            App.logic.addWebsite(companyName, 'https://mock.example.com');
                        }
                    }
                }
                App.ui.renderFileList();
                App.ui.renderWebsiteList();
            },
            addWebsite(companyName = '', url = '') {
                const websiteObject = { id: 'site-' + Date.now() + Math.random(), company: companyName, url: url };
                App.state.websites.push(websiteObject);
                App.ui.renderWebsiteList();
            },
            async startProcessing() {
                if (App.state.isProcessing) return;
                if (App.state.isMockMode) {
                    App.ui.changeStep(3);
                    App.elements.progressText.textContent = '正在載入預覽資料...';
                    setTimeout(() => {
                        App.state.results = MOCK_RESULTS;
                        App.ui.renderResults();
                        App.ui.changeStep(4);
                    }, 1500);
                    return;
                }
                const matchedData = [];
                let hasError = false;
                App.state.websites.forEach(site => {
                    const correspondingFile = App.state.files.find(f => f.name.replace(/\.pdf$/i, '') === site.company);
                    if (!site.company.trim()) { alert(`錯誤：有一個公司名稱欄位是空的。`); hasError = true; return; }
                    if (correspondingFile) {
                        matchedData.push({ file: correspondingFile.file, company: site.company, url: site.url || 'N/A' });
                    }
                });
                if (hasError) return;
                if (matchedData.length === 0) { alert('請至少上傳一份 PDF，並確保其檔名（不含 .pdf）與公司名稱欄位完全相符。'); return; }
                App.state.isProcessing = true;
                App.ui.changeStep(3);
                try {
                    const simpleFormData = new FormData();
                    matchedData.forEach(item => {
                        simpleFormData.append('files', item.file, item.file.name);
                        simpleFormData.append('company_names', item.company);
                        simpleFormData.append('website_urls', item.url);
                    });
                    App.elements.progressText.textContent = `正在準備上傳 ${matchedData.length} 個檔案...`;
                    const response = await fetch('http://127.0.0.1:8000/scoring/batch', { method: 'POST', body: simpleFormData });
                    App.elements.progressText.textContent = '後端處理中，請稍候...';
                    if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.detail || `伺服器錯誤: ${response.status}`); }
                    App.state.results = await response.json();
                    App.ui.renderResults();
                    App.ui.changeStep(4);
                } catch (error) {
                    console.error('處理失敗:', error);
                    alert(`發生錯誤：${error.message}`);
                    App.ui.changeStep(2);
                } finally {
                    App.state.isProcessing = false;
                }
            },
            resetApp() {
                const wasInMockMode = App.state.isMockMode;
                App.state = { ...App.state, currentStep: 1, files: [], websites: [], results: [], isProcessing: false };
                App.ui.renderFileList(); App.ui.renderWebsiteList();
                if (wasInMockMode) { App.logic.exitPreviewMode(); } 
                else { App.ui.changeStep(1); }
            },
            handleFontFile(file) {
                if (file && file.name.toLowerCase().endsWith('.ttf')) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        App.state.fontData = e.target.result;
                        App.elements.fontStatus.textContent = `✅ 已載入字型: ${file.name}`;
                        App.elements.fontStatus.style.color = 'var(--primary-accent)';
                    };
                    reader.onerror = () => {
                        App.elements.fontStatus.textContent = '❌ 字型讀取失敗';
                        App.elements.fontStatus.style.color = 'var(--accent-warn)';
                    };
                    reader.readAsDataURL(file);
                } else { alert('請上傳 .ttf 格式的字型檔案。'); }
            },
            startPreviewMode() {
                App.state.isMockMode = true;
                document.body.classList.add('preview-mode');
                App.elements.previewBanner.style.display = 'block';
            },
            exitPreviewMode() {
                App.state.isMockMode = false;
                document.body.classList.remove('preview-mode');
                App.elements.previewBanner.style.display = 'none';
                App.state.currentStep = 1; App.state.files = []; App.state.websites = []; App.state.results = [];
                App.ui.renderFileList(); App.ui.renderWebsiteList();
                App.ui.updateStepUI();
            },
            async testBackendConnection() {
                const btn = App.elements.testConnectionBtn; const statusDiv = App.elements.connectionStatus;
                const originalText = btn.querySelector('#test-text').textContent;
                btn.disabled = true; btn.querySelector('#test-text').textContent = "測試中...";
                statusDiv.classList.remove('hidden', 'text-green-600', 'text-red-600'); statusDiv.textContent = '';
                try {
                    const response = await fetch('http://127.0.0.1:8000/health');
                    if (response.ok) {
                        const data = await response.json(); statusDiv.textContent = `✅ 連接成功: ${data.message}`;
                        statusDiv.classList.add('text-green-600');
                    } else { throw new Error(`伺服器回應錯誤: ${response.status}`); }
                } catch (error) {
                    statusDiv.textContent = `❌ 連接失敗: ${error.message}. 請確認後端伺服器是否已啟動。`;
                    statusDiv.classList.add('text-red-600');
                } finally {
                    btn.disabled = false; btn.querySelector('#test-text').textContent = originalText;
                    statusDiv.classList.remove('hidden');
                }
            },
        },
        ui: {
            changeStep(step) {
                App.state.currentStep = step;
                this.updateStepUI();
                if (step === 2 && App.state.isMockMode && App.state.files.length === 0) {
                    const mockFile = new File(["mock pdf content"], "永續模範企業 (預覽).pdf", { type: "application/pdf", lastModified: new Date() });
                    App.logic.handleFiles([mockFile]);
                }
            },
            updateStepUI() {
                App.elements.contents.forEach((content, index) => {
                    content.classList.toggle('hidden', (index + 1) !== App.state.currentStep);
                });
                App.elements.stepWrappers.forEach((wrapper, index) => {
                    wrapper.classList.remove('step-active', 'step-completed');
                    if (index + 1 < App.state.currentStep) wrapper.classList.add('step-completed');
                    else if (index + 1 === App.state.currentStep) wrapper.classList.add('step-active');
                });
                App.elements.stepLines.forEach((line, index) => {
                   line.classList.remove('step-line-active', 'step-line-completed');
                   if (index + 1 < App.state.currentStep) line.classList.add('step-line-completed');
                   else if (index + 1 === App.state.currentStep -1) line.classList.add('step-line-active');
                });
            },
            renderFileList() {
                App.elements.fileList.innerHTML = App.state.files.map(f => this.templates.fileItem(f)).join('');
                App.elements.fileList.querySelectorAll('.remove-file-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const fileId = e.currentTarget.dataset.id;
                        App.state.files = App.state.files.filter(file => file.id !== fileId);
                        this.renderFileList();
                    });
                });
            },
            renderWebsiteList() {
                App.elements.websiteList.innerHTML = App.state.websites.map(w => this.templates.websiteItem(w)).join('');
                App.elements.websiteList.querySelectorAll('.remove-website-btn').forEach(btn => {
                   btn.addEventListener('click', (e) => {
                       const siteId = e.currentTarget.dataset.id;
                       App.state.websites = App.state.websites.filter(site => site.id !== siteId);
                       this.renderWebsiteList();
                   });
                });
                App.elements.websiteList.querySelectorAll('input').forEach(input => {
                   input.addEventListener('change', (e) => {
                       const siteId = e.currentTarget.dataset.id; const field = e.currentTarget.dataset.field;
                       const site = App.state.websites.find(s => s.id === siteId);
                       if (site) site[field] = e.target.value;
                   });
                });
            },
            renderResults() {
                App.elements.resultsContainer.innerHTML = App.state.results.map(r => this.templates.resultCard(r)).join('');
                App.state.results.forEach(r => {
                    const companyId = this.templates.sanitizeId(r.company);
                    const exportBtn = document.querySelector(`.export-pdf-btn[data-company-id="${companyId}"]`);
                    if (exportBtn) {
                        exportBtn.addEventListener('click', () => App.pdf.generate(r, exportBtn));
                    }
                    const canvasId = `chart-${companyId}`;
                    const ctx = document.getElementById(canvasId);
                    if(ctx) this.createChart(ctx, r);

                    // 修正：綁定「查看詳細評分」按鈕的點擊事件
                    const toggleBtn = document.getElementById(`toggle-details-${companyId}`);
                    const detailsPanel = document.getElementById(`details-${companyId}`);
                    if (toggleBtn && detailsPanel) {
                        toggleBtn.addEventListener('click', () => {
                            const isHidden = detailsPanel.style.display === 'none';
                            detailsPanel.style.display = isHidden ? 'block' : 'none';
                            toggleBtn.textContent = isHidden ? '隱藏詳細評分' : '查看詳細評分';
                        });
                        detailsPanel.style.display = 'none'; // 預設隱藏
                    }

                    if (detailsPanel) {
                        this.bindDetailViewEvents(detailsPanel);
                    }
                });
            },
            bindDetailViewEvents(container) {
                const navLinks = container.querySelectorAll('.detail-nav-link');
                const contentSections = container.querySelectorAll('.detail-content-section');
                navLinks.forEach(link => {
                    link.addEventListener('click', e => {
                        e.preventDefault();
                        const targetId = link.getAttribute('href').substring(1);
                        document.getElementById(targetId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    });
                });
                const observer = new IntersectionObserver(entries => {
                    entries.forEach(entry => {
                        const id = entry.target.getAttribute('id');
                        const navLink = container.querySelector(`a[href="#${id}"]`);
                        if (entry.isIntersecting) {
                            navLinks.forEach(link => link.classList.remove('bg-blue-100', 'text-primary-blue', 'font-semibold'));
                            navLink?.classList.add('bg-blue-100', 'text-primary-blue', 'font-semibold');
                        }
                    });
                }, { rootMargin: "-40% 0px -60% 0px", threshold: 0 });
                contentSections.forEach(section => observer.observe(section));
            },
            createChart(ctx, result) {
                const reportBreakdown = result.breakdown?.find(b => b.id === 'report');
                const mediaBreakdown = result.breakdown?.find(b => b.id === 'media');
                const labels = [], scores = [], maxScores = [];
                if (reportBreakdown?.sections) {
                    reportBreakdown.sections.forEach(sec => { labels.push(sec.title); scores.push(sec.score || 0); maxScores.push(sec.max_score || 0); });
                }
                if (mediaBreakdown?.sections) {
                    mediaBreakdown.sections.forEach(sec => {
                        labels.push(sec.title);
                        const sectionScore = sec.criteria.reduce((sum, crit) => sum + (crit.score || 0), 0);
                        const sectionMax = sec.criteria.reduce((sum, crit) => sum + crit.max_score, 0);
                        scores.push(sectionScore); maxScores.push(sectionMax);
                    });
                }
                const percentages = scores.map((score, i) => maxScores[i] > 0 ? (score / maxScores[i]) * 100 : 0);
                const chartTextColor = document.body.classList.contains('preview-mode') ? getComputedStyle(document.body).getPropertyValue('--text-secondary') : 'var(--text-secondary)';
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{ label: '得分率 (%)', data: percentages, backgroundColor: 'rgba(53, 99, 124, 0.6)', borderColor: 'rgba(53, 99, 124, 1)', borderWidth: 1, borderRadius: 4 }]
                    },
                    options: {
                        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                        scales: {
                            x: { beginAtZero: true, max: 100, ticks: { color: chartTextColor, callback: (v) => v + "%" }, grid: { color: 'rgba(197, 210, 218, 0.2)' } },
                            y: { ticks: { color: chartTextColor, font: { size: 10 } }, grid: { display: false } }
                        },
                        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.dataset.label || ''}: ${c.parsed.x.toFixed(1)}%` } } }
                    }
                });
            },
            templates: {
                fileItem(file) {
                    return `<div class="flex items-center justify-between bg-gray-50 p-3 rounded-md border" style="border-color: var(--border-color);">
                        <div class="flex items-center gap-3"><svg class="w-6 h-6 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 2a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V8.344a1.996 1.996 0 00-.6-.344l-5-2.5a2 2 0 00-1.932 0l-5 2.5A1.996 1.996 0 004 8.344V4a2 2 0 012-2h4a2 2 0 110 4H6V4a2 2 0 00-2-2z" clip-rule="evenodd" /><path d="M11 11.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" /></svg>
                            <div><p class="text-sm font-medium text-text-primary">${file.name}</p><p class="text-xs text-text-secondary">${(file.size / 1024 / 1024).toFixed(2)} MB</p></div>
                        </div><button data-id="${file.id}" class="remove-file-btn p-1.5 rounded-full hover:bg-gray-200 text-gray-500 hover:text-gray-700"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" /></svg></button>
                    </div>`;
                },
                websiteItem(site) {
                    return `<div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                        <div class="md:col-span-5"><input type="text" data-id="${site.id}" data-field="company" value="${site.company}" placeholder="企業名稱 (須與檔名相同)" class="w-full p-2 border rounded-md" style="border-color: var(--border-color); color: var(--text-primary); background-color: #fff;"></div>
                        <div class="md:col-span-6"><input type="text" data-id="${site.id}" data-field="url" value="${site.url}" placeholder="企業永續網站 URL (選填)" class="w-full p-2 border rounded-md" style="border-color: var(--border-color); color: var(--text-primary); background-color: #fff;"></div>
                        <div class="md:col-span-1 text-right"><button data-id="${site.id}" class="remove-website-btn p-1.5 rounded-full hover:bg-gray-200 text-gray-500 hover:text-gray-700"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clip-rule="evenodd" /></svg></button></div>
                    </div>`;
                },
                resultCard(result) {
                    const companyId = this.sanitizeId(result.company);
                    const level = this.getLevelAndColor(result.totals?.final, true); // Get object with CSS vars for HTML
                    const isPreview = result.company.includes("(預覽)");
                    const mainTitle = result.company.replace(" (預覽)", "");

                    return `<div class="card p-6 md:p-8 shadow-sm border border-gray-200">
                        <div class="flex flex-wrap justify-between items-start gap-4 mb-8">
                            <div class="flex items-center gap-3">
                                <h3 class="text-2xl font-bold text-text-primary">${mainTitle}</h3>
                                ${isPreview ? '<span class="badge-preview">預覽</span>' : ''}
                            </div>
                            <button data-company-id="${companyId}" class="export-pdf-btn btn btn-primary">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                                匯出 PDF 報告
                            </button>
                        </div>
                        <p class="text-md font-normal text-text-secondary -mt-8 mb-8">AI 評分分析報告</p>
                        <div class="grid grid-cols-1 lg:grid-cols-5 gap-8">
                            <div class="lg:col-span-3 space-y-8">
                                <div>
                                    <h4 class="text-base font-semibold text-text-primary">關鍵摘要</h4>
                                    <hr class="mt-3 mb-4" style="border-color: var(--border-color);" />
                                    <div class="space-y-4 max-w-prose leading-7 text-text-secondary">
                                        <p><strong>AI 總評:</strong> ${result.overview_comment || '無'}</p>
                                        <p class="text-xs text-gray-500">2025-10-06 · Gemini v2.5</p>
                                    </div>
                                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
                                        <div class="card p-4 shadow-none border border-gray-200">
                                            <h5 class="font-semibold text-strengths mb-2">主要優勢</h5>
                                            ${this.renderStrengthsImprovements(result.strengths)}
                                        </div>
                                        <div class="card p-4 shadow-none border border-gray-200">
                                            <h5 class="font-semibold text-improvements mb-2">改善建議</h5>
                                            ${this.renderStrengthsImprovements(result.improvements)}
                                        </div>
                                    </div>
                                </div>
                                <div class="border-t pt-8" style="border-color: var(--border-color);">
                                   <button id="toggle-details-${companyId}" class="btn btn-secondary w-full">查看詳細評分</button>
                                </div>
                            </div>
                            <div class="lg:col-span-2 space-y-8">
                               <div class="w-full flex justify-center lg:justify-end">
                                    <div class="text-center p-4 rounded-lg cursor-pointer w-full" style="background-color: ${level.bgColor}; border: 1px solid ${level.borderColor};" title="評分等級說明 (功能開發中)">
                                        <p class="text-4xl font-bold" style="color: ${level.textColor};">${result.totals?.final?.toFixed(1) || '-'}</p>
                                        <p class="text-sm font-semibold" style="color: ${level.textColor};">${level.text}</p>
                                    </div>
                               </div>
                               <div>
                                    <h4 class="text-base font-semibold text-text-primary">分項得分率</h4>
                                    <hr class="mt-3 mb-4" style="border-color: var(--border-color);" />
                                    <div class="chart-container"><canvas id="chart-${companyId}"></canvas></div>
                               </div>
                            </div>
                        </div>
                        <div id="details-${companyId}" class="mt-8 border-t pt-8" style="border-color: var(--border-color);">
                            ${this.renderDetailedView(result, companyId)}
                        </div>
                    </div>`;
                },
                renderStrengthsImprovements(data) {
                   if (!data || Object.keys(data).length === 0) return '<ul><li>無</li></ul>';
                   let html = '';
                   for (const category in data) {
                       if (data[category] && data[category].length > 0) {
                           html += `<div class="mb-3"><strong class="text-sm text-text-primary">${category}</strong><ul class="list-disc pl-5 mt-1 text-sm text-text-secondary space-y-1">`;
                           html += data[category].map(item => `<li>${item}</li>`).join('');
                           html += `</ul></div>`;
                       }
                   }
                   return html || '<ul><li>無</li></ul>';
                },
                renderDetailedView(result, companyId) {
                    return `<div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
                        <!-- Sticky Nav -->
                        <aside class="lg:col-span-1 lg:sticky top-10 self-start">
                            <h5 class="font-semibold mb-3 text-text-primary">評分章節</h5>
                            <ul class="space-y-1">
                                ${(result.breakdown || []).map(block => 
                                    block.sections.map(section => `
                                        <li>
                                            <a href="#section-${this.sanitizeId(section.title)}-${companyId}" class="detail-nav-link flex justify-between items-center p-2 rounded-md hover:bg-gray-100 transition-colors text-sm">
                                                <span>${section.title}</span>
                                                <span class="text-xs font-mono px-1.5 py-0.5 bg-gray-200 rounded">${(section.score||0).toFixed(1)}</span>
                                            </a>
                                        </li>
                                    `).join('')
                                ).join('')}
                            </ul>
                        </aside>
                        <!-- Content -->
                        <main class="lg:col-span-3 space-y-8">
                            ${(result.breakdown || []).map(block => 
                                block.sections.map(section => this.renderSectionCard(section, companyId)).join('')
                            ).join('')}
                        </main>
                    </div>`;
                },
                renderSectionCard(section, companyId) {
                    const pct = section.max_score > 0 ? (section.score || 0) / section.max_score * 100 : 0;
                    return `<div id="section-${this.sanitizeId(section.title)}-${companyId}" class="detail-content-section card p-4 md:p-6 shadow-sm">
                        <div class="flex flex-wrap justify-between items-center gap-2 mb-4">
                            <h6 class="text-base font-semibold detail-view-section-title">${section.title}</h6>
                            <span class="text-sm font-medium text-text-secondary">${(section.score||0).toFixed(1)} / ${section.max_score}</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-1.5 mb-6"><div class="bg-primary-accent h-1.5 rounded-full" style="width: ${pct}%"></div></div>
                        <div class="space-y-4">
                            ${section.criteria.map(c => this.renderCriterion(c)).join('')}
                        </div>
                    </div>`;
                },
                renderCriterion(criterion) {
                    return `<div class="border-b last:border-b-0 pb-3 mb-3" style="border-color: #f0ebe8;">
                        <p class="font-semibold text-primary-accent text-sm">${criterion.title}</p>
                        <div class="mt-2 space-y-3">
                            ${criterion.sub_criteria.map(sc => this.renderSubCriterion(sc)).join('')}
                        </div>
                    </div>`;
                },
                renderSubCriterion(sub) {
                    return `<div class="grid grid-cols-5 gap-4">
                        <div class="col-span-4">
                            <p class="text-sm text-text-primary leading-6 max-w-prose">- ${sub.title}</p>
                            <p class="rationale-text">“${sub.rationale || '無理由'}”</p>
                        </div>
                        <span class="col-span-1 text-sm font-medium text-right text-text-primary whitespace-nowrap">${(sub.score||0).toFixed(1)} / ${sub.max_score}</span>
                    </div>`;
                },
                getLevelAndColor(score, forHTML = false) {
                    const colors = {
                        platinum: { text: '#1f2937', bg: '#f3f4f6', border: '#d1d5db' },
                        gold:     { text: '#854d0e', bg: '#fef9c3', border: '#fde68a' },
                        silver:   { text: '#4b5563', bg: '#e5e7eb', border: '#d1d5db' },
                        copper:   { text: '#9a3412', bg: '#fde6d8', border: '#fed7aa' },
                        default:  { text: '#4b5563', bg: '#e5e7eb', border: '#d1d5db' }
                    };
                    const cssVars = {
                        platinum: { text: 'var(--platinum-text)', bg: 'var(--platinum-bg)' },
                        gold:     { text: 'var(--gold-text)',     bg: 'var(--gold-bg)'     },
                        silver:   { text: 'var(--silver-text)',   bg: 'var(--silver-bg)'   },
                        copper:   { text: 'var(--copper-text)',   bg: 'var(--copper-bg)'   },
                        default:  { text: '#4b5563',              bg: '#e5e7eb'           }
                    };

                    let level, key;
                    if (score == null) { level = '-'; key = 'default'; }
                    else if (score >= 90) { level = '白金級'; key = 'platinum'; }
                    else if (score >= 80) { level = '金級'; key = 'gold'; }
                    else if (score >= 70) { level = '銀級'; key = 'silver'; }
                    else { level = '銅級'; key = 'copper'; }
                    
                    if (forHTML) {
                        return { text: level, textColor: cssVars[key].text, bgColor: cssVars[key].bg, borderColor: colors[key].border };
                    } else {
                        return { text: level, textColor: colors[key].text, bgColor: colors[key].bg, borderColor: colors[key].border };
                    }
                },
                sanitizeId(id) {
                    return (id || '').replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '');
                }
            }
        },
        pdf: {
            async generate(result, buttonElement) {
                const originalButtonHTML = buttonElement.innerHTML;
                buttonElement.innerHTML = `...`; buttonElement.disabled = true;
                try {
                    const { jsPDF } = window.jspdf;
                    const doc = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
                    const hasFont = await this.loadFonts(doc);
                    if (!hasFont) {
                        alert("錯誤：無法產生 PDF，因為中文字型尚未設定。請在第一步驟上傳 .ttf 字型檔。");
                        return; 
                    }
                    
                    doc.outline.add(null, '封面', { pageNumber: 1 });
                    this.createCoverPage(doc, result);
                    
                    doc.addPage();
                    doc.outline.add(null, '總覽', { pageNumber: 2 });
                    await this.createSummaryPage(doc, result);

                    doc.addPage();
                    const detailRoot = doc.outline.add(null, '詳細評分', { pageNumber: 3 });
                    this.createDetailedScorePages(doc, result, detailRoot);
                    
                    const totalPages = doc.internal.getNumberOfPages();
                    for (let i = 1; i <= totalPages; i++) {
                        doc.setPage(i);
                        let chapterTitle = '';
                        if (i === 1) {
                            chapterTitle = '封面';
                        } else if (i === 2) {
                            chapterTitle = '總覽';
                        } else {
                            chapterTitle = '詳細評分';
                        }
                        
                        this._renderHeaderFooter(doc, i, totalPages, result.company.replace(" (預覽)",""), chapterTitle);
                    }

                    const companyName = App.helpers.sanitizeFileName(result.company);
                    doc.save(`${companyName}_AI評分報告_v1.0.pdf`);
                } catch (e) {
                    console.error("PDF generation failed:", e);
                    alert('產生 PDF 時發生錯誤：' + e.message);
                } finally {
                    buttonElement.innerHTML = originalButtonHTML; buttonElement.disabled = false;
                }
            },
            async loadFonts(doc) {
                if (App.state.fontData) {
                    try {
                        const base64Font = App.state.fontData.split(',')[1];
                        doc.addFileToVFS('NotoSansTC-Regular.ttf', base64Font);
                        doc.addFont('NotoSansTC-Regular.ttf', 'NotoSansTC', 'normal');
                        doc.addFileToVFS('NotoSansTC-Bold.ttf', base64Font); // Using same for bold for now
                        doc.addFont('NotoSansTC-Bold.ttf', 'NotoSansTC', 'bold');
                        return true;
                    } catch (e) { console.error("處理 Base64 字型時失敗:", e); return false; }
                } return false;
            },
            _renderHeaderFooter(doc, pageNum, totalPages, companyName, chapterTitle) {
                if (pageNum === 1) return; // No header/footer on cover
                const W = doc.internal.pageSize.width, M = 22, today = new Date().toISOString().split('T')[0];
                doc.setFont('NotoSansTC', 'normal');
                doc.setFontSize(9); App.helpers.setHexColor(doc, '#4b5563'); // gray-600
                doc.text(`${companyName}｜AI 智能評分報告`, M, 18);
                doc.text(chapterTitle, W - M, 18, { align: 'right' });
                doc.setFontSize(8); App.helpers.setHexColor(doc, '#6b7280'); // gray-500
                doc.text(`報告日期 ${today}`, M, 297 - 15);
                doc.text(`Page ${pageNum} of ${totalPages}`, W / 2, 297 - 15, { align: 'center' });
                doc.text(`文件版本 v1.0`, W - M, 297 - 15, { align: 'right' });
            },
            createCoverPage(doc, result) {
                const W = 210, H = 297;
                doc.setFont('NotoSansTC', 'normal');
                doc.setFillColor('#203F58');
                doc.rect(0, H - 30, W, 30, 'F');
                doc.setFontSize(24); App.helpers.setHexColor(doc, '#102030');
                doc.setFont('NotoSansTC', 'bold');
                const mainTitle = result.company.replace(" (預覽)","");
                doc.text(mainTitle, W/2, H/2 - 20, {align: 'center'});
                doc.setFontSize(18); App.helpers.setHexColor(doc, '#35637C');
                doc.setFont('NotoSansTC', 'normal');
                doc.text("AI 智能永續報告書評分報告", W/2, H/2 - 10, {align: 'center'});

                const score = result.totals?.final || 0;
                const level = App.ui.templates.getLevelAndColor(score, false); // Get object with hex codes for PDF
                doc.setLineWidth(2);
                doc.setDrawColor(level.bgColor);
                doc.setFillColor(level.bgColor);
                doc.circle(W/2, H/2 + 40, 36/2, 'F');
                doc.setFontSize(24); App.helpers.setHexColor(doc, level.textColor);
                doc.setFont('NotoSansTC', 'bold');
                doc.text(score.toFixed(1), W/2, H/2 + 38, {align: 'center'});
                doc.setFontSize(10); App.helpers.setHexColor(doc, level.textColor);
                doc.setFont('NotoSansTC', 'normal');
                doc.text(level.text, W/2, H/2 + 46, {align: 'center'});
            },
            async createSummaryPage(doc, result) {
                let y = 32; const M = 22, CONTENT_W = 210 - 2*M;
                doc.setFont('NotoSansTC', 'normal');
                
                doc.setFontSize(18); App.helpers.setHexColor(doc, '#102030');
                doc.setFont('NotoSansTC', 'bold');
                doc.text("Executive Snapshot / 總覽", M, y); y += 12;

                doc.setFontSize(11); App.helpers.setHexColor(doc, '#1e293b');
                doc.text("此頁面為報告總覽，詳細評分請見後續章節。", M, y);
            },
            createDetailedScorePages(doc, result, parentBookmark) {
                const W = 210, H = 297, M = 22;
                let y = 32;

                const checkBreak = (needed) => {
                    if (y + needed > H - 28) {
                        doc.addPage();
                        y = 32;
                    }
                };
                
                for (const block of (result.breakdown || [])) {
                    const blockTitle = block.id === 'report' ? '永續報告書' : '多元媒體應用';
                    doc.outline.add(parentBookmark, blockTitle, { pageNumber: doc.internal.getCurrentPageInfo().pageNumber });

                    for (const section of (block.sections || [])) {
                        checkBreak(20);
                        doc.setFontSize(14); doc.setFont('NotoSansTC', 'bold'); App.helpers.setHexColor(doc, '#102030');
                        doc.text(section.title, M, y);
                        doc.setLineWidth(0.2); doc.setDrawColor(200); doc.line(M, y + 2, W-M, y + 2);
                        y += 10;
                        
                        for (const criterion of (section.criteria || [])) {
                            checkBreak(15);
                            doc.setFontSize(12); doc.setFont('NotoSansTC', 'bold'); App.helpers.setHexColor(doc, '#35637C');
                            doc.text(criterion.title, M, y);
                            y += 8;

                            for (const sub of (criterion.sub_criteria || [])) {
                                doc.setFont('NotoSansTC', 'normal');
                                const scoreText = `${(sub.score || 0).toFixed(1)} / ${sub.max_score}`;
                                const titleText = `- ${sub.title}`;
                                const titleLines = doc.splitTextToSize(titleText, 125);
                                const rationaleText = sub.rationale ? `“${sub.rationale.replace(/p\.(?=\d)/g, 'p. ')}”` : '“無理由”'; // Add space for p.##
                                
                                const evidenceMatch = rationaleText.match(/(p\.\s?\d+|表\s?\d+-\d+)/);
                                const evidenceText = evidenceMatch ? evidenceMatch[0].replace(/\s/g, '') : '';
                                
                                const rationaleLines = doc.splitTextToSize(rationaleText.replace(evidenceMatch?.[0] || '', ''), 125);
                                
                                const needed = (titleLines.length + rationaleLines.length) * 5 + 4;
                                checkBreak(needed);

                                App.helpers.setHexColor(doc, '#1e293b'); doc.setFontSize(11);
                                doc.text(titleLines, M + 5, y);
                                
                                const titleHeight = doc.getTextDimensions(titleLines).h;
                                doc.text(scoreText, W - M, y, { align: 'right' });
                                y += titleHeight + 1;

                                App.helpers.setHexColor(doc, '#475569'); doc.setFontSize(9);
                                doc.text(rationaleLines, M + 5, y);
                                
                                if (evidenceText) {
                                    doc.setFillColor(238, 242, 255); // indigo-50
                                    const evidenceWidth = doc.getTextWidth(evidenceText) + 4;
                                    const rationaleHeight = doc.getTextDimensions(rationaleLines).h;
                                    doc.roundedRect(W - M - evidenceWidth, y + rationaleHeight - 4, evidenceWidth, 5, 1, 1, 'F');
                                    App.helpers.setHexColor(doc, '#4338ca'); // indigo-700
                                    doc.text(evidenceText, W - M - 2, y + rationaleHeight, { align: 'right' });
                                }

                                y += doc.getTextDimensions(rationaleLines).h + 3;
                            }
                            y += 5;
                        }
                    }
                }
            }
        },
        helpers: {
            setHexColor(doc, hex) {
                const r = parseInt(hex.substring(1, 3), 16);
                const g = parseInt(hex.substring(3, 5), 16);
                const b = parseInt(hex.substring(5, 7), 16);
                doc.setTextColor(r, g, b);
            },
            sanitizeFileName(name) { return (name || 'report').replace(/[\\/:*?"<>|\n\r]+/g, '_').slice(0, 128); },
            preloadDefaultFont() {
                if (PRELOADED_FONT_BASE64 && PRELOADED_FONT_BASE64.startsWith('data:font/ttf;base64,')) {
                    App.state.fontData = PRELOADED_FONT_BASE64;
                    App.elements.fontStatus.textContent = `✅ 已載入預設字型`;
                    App.elements.fontStatus.style.color = 'var(--primary-accent)';
                    App.elements.fontUploadBtn.textContent = '更換 PDF 中文字型';
                }
            }
        }
    };

    const MOCK_RESULTS = [
      {
        "company": "永續模範企業 (預覽)",
        "overview_comment": "永續模範企業在各個面向都展現了卓越的領導力。報告書結構清晰、數據透明，並有效利用多元媒體進行溝通，是業界的標竿。(此為預覽模式)",
        "strengths": { "完整性": ["重大性議題矩陣圖呈現清晰，且各章節均能有效連結回重大議題。", "策略面完整揭露了短、中、長期的永續發展藍圖與目標。"], "可信度": ["附有會計師出具的合理等級確信報告，大幅提升資訊可信度。", "治理結構清晰，已揭露高階薪酬與 ESG 績效的連結性。"], "多元媒體應用": ["官方網站設有內容豐富的永續專區，且更新頻繁、資訊即時。", "網站中包含多部高品質的永續專題影片，溝通方式多元。"] },
        "improvements": { "溝通性": ["報告書架構雖清晰，但可考慮增加互動式導覽功能，提升讀者查閱體驗。", "部分利害關係人議合的回應較為制式，可再強化客製化與深度。"], "可信度": ["雖已揭露多數績效，但可考慮增加過去負面訊息的說明與改善措施，展現更高的透明度。"] },
        "breakdown": [
          {
            "id": "report", "score": 92.0, "max_score": 100,
            "sections": [
              {"title": "完整性", "score": 38.0, "max_score": 40.0, "criteria": [
                  {"title": "重大性議題", "score": 8.0, "max_score": 8.0, "sub_criteria": [ {"title": "是否清楚列出或呈現重大性議題分析之矩陣圖或其他圖表，且清楚標明各項議題的種類", "max_score": 2.0, "score": 2.0, "rationale": "報告書第XX頁呈現了清晰的重大性議題矩陣圖。"}, {"title": "是否清楚說明組織重大性議題分析之過程與方法", "max_score": 2.0, "score": 2.0, "rationale": "有詳細說明分析過程。"}, {"title": "是否有呈現出重大性議題在報告書中的連結性", "max_score": 2.0, "score": 2.0, "rationale": "各章節均能連結回重大議題。"}, {"title": "是否清楚說明重大性議題對於組織的意義", "max_score": 2.0, "score": 2.0, "rationale": "說明了議題對營運的影響。"} ]},
                  {"title": "利害關係人共融", "score": 6.0, "max_score": 6.0, "sub_criteria": [ {"title": "是否清楚列出組織的利害關係人之種類與意義", "max_score": 1.0, "score": 1.0, "rationale": "已鑑別主要利害關係人。"}, {"title": "是否清楚說明各種利害關係人議合之方法", "max_score": 2.0, "score": 2.0, "rationale": "提供了多元的溝通管道列表。"}, {"title": "是否清楚說明各種利害關係人關注之議題", "max_score": 1.0, "score": 1.0, "rationale": "已歸納各方關注焦點。"}, {"title": "是否清楚說明組織針對各項議題的因應之道", "max_score": 2.0, "score": 2.0, "rationale": "針對關注議題提出具體回應。"} ]},
                  {"title": "策略", "score": 12.0, "max_score": 12.0, "sub_criteria": [ {"title": "報告書中是否有說明永續對組織的重要性與意義(價值鏈的呈現)", "max_score": 2.0, "score": 2.0, "rationale": "開篇即闡述永續價值主張。"}, {"title": "報告書中是否有揭露組織營運相關之內外部環境分析", "max_score": 3.0, "score": 3.0, "rationale": "包含 PESTEL 分析。"}, {"title": "報告書中是否有揭露組織對於環境、社會、治理等面向的發展原則與管理機制(長期策略)", "max_score": 3.0, "score": 3.0, "rationale": "揭露了2030永續藍圖。"}, {"title": "是否有在各個面向或是各類重大性議題說明組織未來改善目標(中期策略)", "max_score": 2.0, "score": 2.0, "rationale": "設定了具體的3年改善目標。"}, {"title": "針對各項重大性議題是否有設定隔年度之量化或是質化目標(短期策略)", "max_score": 2.0, "score": 2.0, "rationale": "各章節末皆有次年度KPI。"} ]},
                  {"title": "組織介紹", "score": 2.0, "max_score": 2.0, "sub_criteria": [ {"title": "揭露資訊：主要產品與服務、財務績效、地理分布、員工資訊、整體環境與組織營運之關聯性等", "max_score": 2.0, "score": 2.0, "rationale": "公司概況章節完整揭露。"} ]},
                  {"title": "重大永續規範執行及資訊揭露", "score": 10.0, "max_score": 12.0, "sub_criteria": [ {"title": "氣候相關財務揭露(TCFD)", "max_score": 3.0, "score": 3.0, "rationale": "設有獨立 TCFD 報告書章節。"}, {"title": "永續會計準則委員會準則(SASB)", "max_score": 3.0, "score": 3.0, "rationale": "提供 SASB 指標索引。"}, {"title": "自然相關財務揭露(TNFD)", "max_score": 3.0, "score": 2.0, "rationale": "已初步導入 TNFD 框架進行評估。"}, {"title": "國際財務報導準則(IFRS) S1,S2揭露", "max_score": 3.0, "score": 2.0, "rationale": "已開始進行IFRS S1/S2的鑑別與準備。"} ]}
              ]},
              {"title": "可信度", "score": 32.5, "max_score": 35.0, "criteria": [
                  {"title": "管理流程", "score": 10.0, "max_score": 10.0, "sub_criteria": [ {"title": "報告揭露採用之指引與準則", "max_score": 1.0, "score": 1.0, "rationale": "明確遵循 GRI Standards。"}, {"title": "是否揭露報告書主要負責單位", "max_score": 1.0, "score": 1.0, "rationale": "載明永續發展委員會為負責單位。"}, {"title": "報告書的管理方式", "max_score": 4.0, "score": 4.0, "rationale": "說明了資料收集與審核流程。"}, {"title": "針對各項重大性議題皆說明管理方針", "max_score": 4.0, "score": 4.0, "rationale": "各議題均有對應的管理方針。"} ]},
                  {"title": "利害關係人回應", "score": 4.5, "max_score": 5.0, "sub_criteria": [ {"title": "針對利害關係人關注之議題，組織是否實際回應議題，並提出相對應之作為、策略與規劃等政策", "max_score": 2.0, "score": 2.0, "rationale": "均有對應的回應與作為。"}, {"title": "組織是否有針對組織鑑別出之實質性議題進行回應，並提出相對應之策略與作為", "max_score": 3.0, "score": 2.5, "rationale": "大部分議題有回應，少部分較為籠統。"} ]},
                  {"title": "治理", "score": 4.0, "max_score": 5.0, "sub_criteria": [ {"title": "是否有說明組織組織針對永續報告的責任單位", "max_score": 1.0, "score": 1.0, "rationale": "已說明責任單位。"}, {"title": "報告書是否有說明董事會的薪酬與永續績效的連結性", "max_score": 2.0, "score": 2.0, "rationale": "已揭露高階薪酬與ESG 績效連結。"}, {"title": "報告書是否有揭露組織組織的風險與可能之機會(因應之道)", "max_score": 1.0, "score": 1.0, "rationale": "風險管理章節有說明。"}, {"title": "組織績效指標管理方針是否與組織永續原則一致", "max_score": 1.0, "score": 0.0, "rationale": "未明確說明其一致性。"} ]},
                  {"title": "績效", "score": 4.0, "max_score": 5.0, "sub_criteria": [ {"title": "績效之揭露是否完整(重大性議題涵蓋經濟、環境與社會，是否有質化的說明與數據)", "max_score": 2.0, "score": 2.0, "rationale": "績效數據揭露完整。"}, {"title": "重大性議題是否有量化的圖表說明", "max_score": 1.0, "score": 1.0, "rationale": "多數採用量化圖表。"}, {"title": "是否有揭露過去負面訊息", "max_score": 1.0, "score": 0.0, "rationale": "未見揭露負面訊息。"}, {"title": "績效的呈現是否易懂", "max_score": 1.0, "score": 1.0, "rationale": "圖表清晰易懂。"} ]},
                  {"title": "保證/確信", "score": 10.0, "max_score": 10.0, "sub_criteria": [ {"title": "是否已建立永續資訊編制內部控制制度及相關流程", "max_score": 2.0, "score": 2.0, "rationale": "已建立相關內控制度。"}, {"title": "永續資訊編制內部控制制度及其內部稽核執行情形說明", "max_score": 3.0, "score": 3.0, "rationale": "說明了內稽的執行狀況。"}, {"title": "是否有外部第三方獨立保證/確信之佐證資料", "max_score": 2.0, "score": 2.0, "rationale": "附有會計師出具之確信報告。"}, {"title": "外部保證是否有說明保證等級、範疇與方法(中度/有限等級者最多得2分，高度/合理等級者最多可得3分)", "max_score": 3.0, "score": 3.0, "rationale": "提供了合理等級的確信。"} ]}
              ]},
              {"title": "溝通性", "score": 21.5, "max_score": 25.0, "criteria": [
                  {"title": "展現", "score": 9.0, "max_score": 10.0, "sub_criteria": [ {"title": "版面是否圖表與文字說明比例恰當，內容清晰且易於閱讀", "max_score": 3.0, "score": 3.0, "rationale": "圖文並茂，排版優良。"}, {"title": "具有英文版報告書", "max_score": 3.0, "score": 3.0, "rationale": "提供完整的英文版報告書。"}, {"title": "展現創新的資訊呈現方式", "max_score": 2.0, "score": 1.0, "rationale": "有使用資訊圖表，但較少創新互動設計。"}, {"title": "報告書之份量是否適當(頁數120-150頁為參考範圍)", "max_score": 2.0, "score": 2.0, "rationale": "頁數適中(130頁)。"} ]},
                  {"title": "利害關係人共融", "score": 4.5, "max_score": 5.0, "sub_criteria": [ {"title": "組織永續報告書是否公開下載", "max_score": 1.0, "score": 1.0, "rationale": "官網提供公開下載。"}, {"title": "是否有說明利害關係人議合(溝通資訊)的方法", "max_score": 2.0, "score": 2.0, "rationale": "已說明溝通方法。"}, {"title": "利害關係人議合的結果，組織是否公開揭露其相對應的回應與作為", "max_score": 2.0, "score": 1.5, "rationale": "有揭露，但部分回應較為制式。"} ]},
                  {"title": "架構", "score": 8.0, "max_score": 10.0, "sub_criteria": [ {"title": "是否清楚整理並呈現本年度的亮點作為報告書的總結", "max_score": 3.0, "score": 3.0, "rationale": "報告書前段有 Highlight 整理。"}, {"title": "完整的索引設計(包括GRI, SASB及其他重要規範等)", "max_score": 3.0, "score": 3.0, "rationale": "附錄提供完整 GRI/SASB 索引。"}, {"title": "報告書附有清楚的連結，使讀者可透過網頁的說明獲得更細節的資訊", "max_score": 2.0, "score": 1.0, "rationale": "部分連結可點擊，但並非全部。"}, {"title": "架構呈現完整易於查閱", "max_score": 2.0, "score": 1.0, "rationale": "目錄清晰，但缺乏互動式導覽。"} ]}
              ]}
            ]
          },
          {
            "id": "media", "score": 18.0, "max_score": 19.0,
            "sections": [
              {"title": "多元媒體應用及內容品質", "score": 18.0, "max_score": 19.0, "criteria": [
                  {"title": "組織永續專區", "score": 3.0, "max_score": 3.0, "sub_criteria": [ {"title": "設置組織永續專區", "max_score": 0.5, "score": 0.5, "rationale": "官網設有永續專區。"}, {"title": "是否將組織永續專區連結設於首頁", "max_score": 0.5, "score": 0.5, "rationale": "首頁有明顯連結。"}, {"title": "是否提供報告書下載", "max_score": 0.5, "score": 0.5, "rationale": "提供歷年報告書下載。"}, {"title": "是否有網站地圖", "max_score": 0.5, "score": 0.5, "rationale": "網站頁尾提供網站地圖。"}, {"title": "站內搜尋引擎", "max_score": 0.5, "score": 0.5, "rationale": "具備站內搜尋功能。"}, {"title": "是否將組織永續專區分類且內容充實", "max_score": 0.5, "score": 0.5, "rationale": "內容分類清晰，資訊豐富。"} ]},
                  {"title": "網頁管理與即時更新", "score": 4.0, "max_score": 4.0, "sub_criteria": [ {"title": "判斷依據：由最新消息觀察網頁是否為最新訊息、是否即時更新", "max_score": 4.0, "score": 4.0, "rationale": "最新消息更新頻繁，資訊即時。"} ]},
                  {"title": "電子版報告書與關鍵資訊連結", "score": 3.0, "max_score": 4.0, "sub_criteria": [ {"title": "按照永續報告定義，須符合環境、社會與治理(ESG)以及供應鏈管理等四項議題之揭露", "max_score": 4.0, "score": 3.0, "rationale": "網站內容涵蓋ESG各面向。"} ]},
                  {"title": "多元媒體展現", "score": 4.0, "max_score": 4.0, "sub_criteria": [ {"title": "文字說明", "max_score": 1.0, "score": 1.0, "rationale": "文字清晰易懂。"}, {"title": "圖表說明", "max_score": 1.0, "score": 1.0, "rationale": "使用多樣化的互動圖表。"}, {"title": "使用影片", "max_score": 1.0, "score": 1.0, "rationale": "網站中包含多部高品質的永續專題影片。"}, {"title": "互動式網頁", "max_score": 1.0, "score": 1.0, "rationale": "提供了互動式的數據查詢頁面。"} ]},
                  {"title": "溝通回饋管道與社群網絡互動", "score": 4.0, "max_score": 4.0, "sub_criteria": [ {"title": "線上回饋機制之應用(網路填寫或連結至電子信箱)", "max_score": 1.0, "score": 1.0, "rationale": "提供線上聯絡表單。"}, {"title": "線上互動式機制之應用", "max_score": 1.0, "score": 1.0, "rationale": "設有利害關係人專區。"}, {"title": "社交網站之應用", "max_score": 1.0, "score": 1.0, "rationale": "活躍於 LinkedIn, Facebook 等社群平台。"}, {"title": "提供訂閱電子報", "max_score": 1.0, "score": 1.0, "rationale": "提供永續電子報訂閱服務。"} ]}
              ]}
            ]
          }
        ],
        "totals": { "report": 55.2, "media": 37.9, "final": 93.1 }
      }
    ];

    App.init();
});


