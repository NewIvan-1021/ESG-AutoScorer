document.addEventListener('DOMContentLoaded', () => {

    // --- 應用程式設定 ---
    const Config = {
        API_BASE_URL: 'http://127.0.0.1:8000',
        FONT_PATHS: {
            // **修正**：只保留 regular 字體的路徑
            normal: '/static/Font/NotoSansTC-Regular.ttf',
        },
        COVER_IMAGE_BASE64: "", // 可選：貼上封面圖片的 Base64
    };

    // --- 輔助工具模組 ---
    const Helpers = {
        /**
         * 從 URL 非同步載入字型檔案，並將其轉換為 Base64。
         * @param {string} url - 字型檔案的 URL 路徑。
         * @returns {Promise<string|null>} - Base64 編碼的字型數據，或在失敗時返回 null。
         */
        loadFontAsBase64: async (url) => {
            try {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`網路回應錯誤，狀態碼: ${response.status}`);
                const fontBlob = await response.blob();
                const reader = new FileReader();
                return new Promise((resolve, reject) => {
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(fontBlob);
                });
            } catch (error) {
                console.error(`字型載入失敗: ${url}`, error);
                // 將錯誤物件本身也回傳，以便提供更詳細的資訊
                return { error: `無法從 ${url} 載入字型。請檢查後端 CORS 設定與檔案路徑。錯誤詳情: ${error.message}` };
            }
        },

        /**
         * 將十六進位顏色碼轉換為 jsPDF 可用的 RGB 陣列。
         * @param {jsPDF} doc - jsPDF 實例。
         * @param {string} hex - 十六進位顏色碼 (例如 "#RRGGBB")。
         * @param {'text'|'draw'|'fill'} [type='text'] - 應用顏色的類型。
         */
        setHexColor: (doc, hex, type = 'text') => {
            const r = parseInt(hex.substring(1, 3), 16);
            const g = parseInt(hex.substring(3, 5), 16);
            const b = parseInt(hex.substring(5, 7), 16);
            if (type === 'text') doc.setTextColor(r, g, b);
            else if (type === 'draw') doc.setDrawColor(r, g, b);
            else if (type === 'fill') doc.setFillColor(r, g, b);
        },

        /**
         * 清理檔案名稱，移除不合法的字元。
         * @param {string} name - 原始檔案名稱。
         * @returns {string} - 清理後的檔案名稱。
         */
        sanitizeFileName: (name) => (name || 'report').replace(/[\\/:*?"<>|\n\r]+/g, '_').slice(0, 128),
    };

    // --- UI 介面管理模組 ---
    const UI = {
        elements: {}, // 將在 init 中填充
        
        init(appState) {
            this.state = appState;
            this.elements = {
                // ... (從 App 移至此處)
                steps: [document.getElementById('step1'), document.getElementById('step2'), document.getElementById('step3'), document.getElementById('step4')],
                stepWrappers: [document.getElementById('step-wrapper-1'), document.getElementById('step-wrapper-2'), document.getElementById('step-wrapper-3'), document.getElementById('step-wrapper-4')],
                stepLines: [document.getElementById('step-line-1'), document.getElementById('step-line-2'), document.getElementById('step-line-3')],
                contents: [document.getElementById('step1-content'), document.getElementById('step2-content'), document.getElementById('step3-content'), document.getElementById('step4-content')],
                fileList: document.getElementById('file-list'),
                websiteList: document.getElementById('website-list'),
                resultsContainer: document.getElementById('results-container'),
                previewBanner: document.getElementById('preview-banner'),
                fontStatus: document.getElementById('font-status'),
            };
        },

        /**
         * 切換到指定的步驟。
         * @param {number} step - 要切換到的步驟編號。
         */
        changeStep(step) {
            if (step === 2 && !this.state.fontData.normal && !this.state.isMockMode) {
                alert('預設字型仍在載入中或載入失敗，請稍後再試。');
                return;
            }
            this.state.currentStep = step;
            this.updateStepUI();
            if (step === 2 && this.state.isMockMode && this.state.files.length === 0) {
                const mockFile = new File(["mock pdf content"], "永續模範企業 (預覽).pdf", { type: "application/pdf", lastModified: new Date() });
                Logic.handleFiles([mockFile]);
            }
        },

        /**
         * 根據當前步驟更新進度條的 UI 狀態。
         */
        updateStepUI() {
            this.elements.contents.forEach((content, index) => {
                content.classList.toggle('hidden', (index + 1) !== this.state.currentStep);
            });
            this.elements.stepWrappers.forEach((wrapper, index) => {
                wrapper.classList.remove('step-active', 'step-completed');
                if (index + 1 < this.state.currentStep) wrapper.classList.add('step-completed');
                else if (index + 1 === this.state.currentStep) wrapper.classList.add('step-active');
            });
            this.elements.stepLines.forEach((line, index) => {
                line.classList.remove('step-line-active', 'step-line-completed');
                if (index + 1 < this.state.currentStep) line.classList.add('step-line-completed');
                else if (index + 1 === this.state.currentStep - 1) line.classList.add('step-line-active');
            });
        },
        
        // ... (renderFileList, renderWebsiteList, renderResults 等渲染函數)
        renderFileList() {
            this.elements.fileList.innerHTML = this.state.files.map(f => this.templates.fileItem(f)).join('');
            this.elements.fileList.querySelectorAll('.remove-file-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const fileId = e.currentTarget.dataset.id;
                    this.state.files = this.state.files.filter(file => file.id !== fileId);
                    this.renderFileList();
                });
            });
        },

        renderWebsiteList() {
            this.elements.websiteList.innerHTML = this.state.websites.map(w => this.templates.websiteItem(w)).join('');
            this.elements.websiteList.querySelectorAll('.remove-website-btn').forEach(btn => {
               btn.addEventListener('click', (e) => {
                   const siteId = e.currentTarget.dataset.id;
                   this.state.websites = this.state.websites.filter(site => site.id !== siteId);
                   this.renderWebsiteList();
               });
            });
            this.elements.websiteList.querySelectorAll('input').forEach(input => {
               input.addEventListener('change', (e) => {
                   const siteId = e.currentTarget.dataset.id; const field = e.currentTarget.dataset.field;
                   const site = this.state.websites.find(s => s.id === siteId);
                   if (site) site[field] = e.target.value;
               });
            });
        },

        renderResults() {
            this.elements.resultsContainer.innerHTML = this.state.results.map(r => this.templates.resultCard(r)).join('');
            this.state.results.forEach(r => {
                const companyId = this.templates.sanitizeId(r.company);
                const exportBtn = document.querySelector(`.export-pdf-btn[data-company-id="${companyId}"]`);
                if (exportBtn) {
                    exportBtn.addEventListener('click', () => PDFGenerator.generate(r, exportBtn));
                }
                const canvasId = `chart-${companyId}`;
                const ctx = document.getElementById(canvasId);
                if(ctx) this.createChart(ctx, r);

                const toggleBtn = document.getElementById(`toggle-details-${companyId}`);
                const detailsPanel = document.getElementById(`details-${companyId}`);
                if (toggleBtn && detailsPanel) {
                    toggleBtn.addEventListener('click', () => {
                        const isHidden = detailsPanel.style.display === 'none';
                        detailsPanel.style.display = isHidden ? 'block' : 'none';
                        toggleBtn.textContent = isHidden ? '隱藏詳細評分' : '查看詳細評分';
                    });
                    detailsPanel.style.display = 'none';
                }

                if (detailsPanel) {
                    this.bindDetailViewEvents(detailsPanel);
                }
            });
        },
        
        bindDetailViewEvents(container) {
            // ... (與之前相同)
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
        
        _getChartConfig(result) {
            // ... (與之前相同)
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
            const chartTextColor = '#4b5563';
            
            return {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{ label: '得分率 (%)', data: percentages, backgroundColor: 'rgba(53, 99, 124, 0.6)', borderColor: 'rgba(53, 99, 124, 1)', borderWidth: 1, borderRadius: 4 }]
                },
                options: {
                    indexAxis: 'y', responsive: false, maintainAspectRatio: true,
                    animation: false,
                    scales: {
                        x: { beginAtZero: true, max: 100, ticks: { color: chartTextColor, callback: (v) => v + "%" }, grid: { color: 'rgba(197, 210, 218, 0.2)' } },
                        y: { ticks: { color: chartTextColor, font: { size: 14 } }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false }, tooltip: { enabled: false } },
                    layout: { padding: { left: 10, right: 10, top: 5, bottom: 5 } }
                }
            };
        },
        
        createChart(ctx, result) {
            // ... (與之前相同)
            const config = this._getChartConfig(result);
            config.options.responsive = true;
            config.options.maintainAspectRatio = false;
            config.options.animation = true;
            config.options.plugins.tooltip.enabled = true;
            config.options.plugins.tooltip.callbacks = { label: (c) => `${c.dataset.label || ''}: ${c.parsed.x.toFixed(1)}%` };
            config.options.scales.y.ticks.font.size = 10;
            
            const chartTextColor = document.body.classList.contains('preview-mode') ? getComputedStyle(document.body).getPropertyValue('--text-secondary') : 'var(--text-secondary)';
            config.options.scales.x.ticks.color = chartTextColor;
            config.options.scales.y.ticks.color = chartTextColor;
            new Chart(ctx, config);
        },

        templates: {
             // ... (所有 template 函數移至此處)
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
                const level = this.getLevelAndColor(result.totals?.final, true);
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
    };

    // --- PDF 生成模組 ---
    const PDFGenerator = {
        state: null,

        init(appState) {
            this.state = appState;
        },

        async generate(result, buttonElement) {
            const originalButtonHTML = buttonElement.innerHTML;
            buttonElement.disabled = true;
            buttonElement.innerHTML = `<svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>正在產生...`;
            
            try {
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
                
                if (!this.state.fontData.normal || !this.state.fontData.bold) {
                    alert("錯誤：無法產生 PDF，因為必要的中文字型載入失敗。");
                    return; 
                }
                this.loadFontsToVFS(doc);

                doc.outline.add(null, '封面', { pageNumber: 1 });
                await this.createCoverPage(doc, result);
                
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
                    if (i === 1) chapterTitle = '封面';
                    else if (i === 2) chapterTitle = '總覽';
                    else chapterTitle = '詳細評分';
                    this._renderHeaderFooter(doc, i, totalPages, result.company.replace(" (預覽)",""), chapterTitle);
                }

                const companyName = Helpers.sanitizeFileName(result.company);
                doc.save(`${companyName}_AI評分報告_v1.0.pdf`);
            } catch (e) {
                console.error("PDF generation failed:", e);
                alert('產生 PDF 時發生錯誤：' + e.message);
            } finally {
                buttonElement.innerHTML = originalButtonHTML;
                buttonElement.disabled = false;
            }
        },

        loadFontsToVFS(doc) {
            doc.addFileToVFS('NotoSansTC-Regular.ttf', this.state.fontData.normal);
            doc.addFont('NotoSansTC-Regular.ttf', 'NotoSansTC', 'normal');
            doc.addFileToVFS('NotoSansTC-Bold.ttf', this.state.fontData.bold);
            doc.addFont('NotoSansTC-Bold.ttf', 'NotoSansTC', 'bold');
        },
        
        // ... (所有 PDF 頁面創建函數，如 createCoverPage, createSummaryPage 等)
        _renderHeaderFooter(doc, pageNum, totalPages, companyName, chapterTitle) {
            if (pageNum === 1) return;
            const W = doc.internal.pageSize.width, M = 22, today = new Date().toISOString().split('T')[0];
            doc.setFont('NotoSansTC', 'normal');
            doc.setFontSize(9); Helpers.setHexColor(doc, '#4b5563');
            doc.text(`${companyName}｜AI 智能評分報告`, M, 18);
            doc.text(chapterTitle, W - M, 18, { align: 'right' });
            doc.setFontSize(8); Helpers.setHexColor(doc, '#6b7280');
            doc.text(`報告日期 ${today}`, M, 297 - 15);
            doc.text(`Page ${pageNum} of ${totalPages}`, W / 2, 297 - 15, { align: 'center' });
            doc.text(`文件版本 v1.0`, W - M, 297 - 15, { align: 'right' });
        },
        async createCoverPage(doc, result) {
            const W = 210, H = 297;
            doc.setFont('NotoSansTC', 'normal');

            if (Config.COVER_IMAGE_BASE64) {
                try {
                    const imgProps = doc.getImageProperties(Config.COVER_IMAGE_BASE64);
                    const imgRatio = imgProps.width / imgProps.height;
                    const pageRatio = W / H;
                    let imgWidth, imgHeight, startX, startY;

                    if (imgRatio > pageRatio) {
                        imgWidth = W;
                        imgHeight = W / imgRatio;
                        startX = 0;
                        startY = (H - imgHeight) / 2;
                    } else {
                        imgHeight = H;
                        imgWidth = H * imgRatio;
                        startY = 0;
                        startX = (W - imgWidth) / 2;
                    }
                    doc.addImage(Config.COVER_IMAGE_BASE64, 'PNG', startX, startY, imgWidth, imgHeight);
                } catch (e) {
                    console.error("無法加入封面圖片:", e);
                    doc.setFillColor('#203F58');
                    doc.rect(0, 0, W, H, 'F');
                }
            } else {
                doc.setFillColor('#203F58');
                doc.rect(0, H - 30, W, 30, 'F');
            }

            doc.setFontSize(24); Helpers.setHexColor(doc, '#102030');
            doc.setFont('NotoSansTC', 'bold');
            const mainTitle = result.company.replace(" (預覽)","");
            doc.text(mainTitle, W/2, H/2 - 20, {align: 'center'});
            doc.setFontSize(18); Helpers.setHexColor(doc, '#35637C');
            doc.setFont('NotoSansTC', 'normal');
            doc.text("AI 智能永續報告書評分報告", W/2, H/2 - 10, {align: 'center'});

            const score = result.totals?.final || 0;
            const level = UI.templates.getLevelAndColor(score, false);
            doc.setLineWidth(2);
            doc.setDrawColor(level.bgColor);
            doc.setFillColor(level.bgColor);
            doc.circle(W/2, H/2 + 40, 36/2, 'F');
            doc.setFontSize(24); Helpers.setHexColor(doc, level.textColor);
            doc.setFont('NotoSansTC', 'bold');
            doc.text(score.toFixed(1), W/2, H/2 + 38, {align: 'center'});
            doc.setFontSize(10); Helpers.setHexColor(doc, level.textColor);
            doc.setFont('NotoSansTC', 'normal');
            doc.text(level.text, W/2, H/2 + 46, {align: 'center'});
        },
        async _getChartImage(result) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const config = UI._getChartConfig(result);

            const numLabels = config.data.labels.length;
            canvas.width = 800;
            canvas.height = Math.max(250, numLabels * 35 + 50);
            config.options.scales.y.ticks.font.size = 12;

            return new Promise((resolve) => {
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                config.options.animation = {
                    onComplete: () => {
                        setTimeout(() => resolve(canvas.toDataURL('image/png')), 50);
                    }
                };
                new Chart(ctx, config);
            });
        },
        drawCategoryList(doc, data, x, startY, maxWidth) {
            let y = startY;
            const lineHeight = 1.6;

            if (!data || Object.keys(data).length === 0) {
                doc.setFontSize(9);
                doc.setFont('NotoSansTC', 'normal');
                Helpers.setHexColor(doc, '#475569');
                doc.text("• 無", x, y);
                return y + 7;
            }

            for (const category in data) {
                if (data[category] && data[category].length > 0) {
                    doc.setFontSize(10);
                    doc.setFont('NotoSansTC', 'bold');
                    Helpers.setHexColor(doc, '#1e293b');
                    
                    const catLines = doc.splitTextToSize(category, maxWidth);
                    doc.text(catLines, x, y);
                    y += doc.getTextDimensions(catLines).h + 2;

                    doc.setFontSize(9);
                    doc.setFont('NotoSansTC', 'normal');
                    Helpers.setHexColor(doc, '#475569');

                    for (const item of data[category]) {
                        const itemLines = doc.splitTextToSize(item, maxWidth - 5);
                        doc.text("•", x, y + 3.5);
                        doc.text(itemLines, x + 5, y + 3.5, { lineHeightFactor: lineHeight });
                        y += doc.getTextDimensions(itemLines, { lineHeightFactor: lineHeight }).h + 3;
                    }
                    y += 4;
                }
            }
            return y;
        },
        async createSummaryPage(doc, result) {
            let y = 32; const M = 22, CONTENT_W = 210 - 2*M;
            doc.setFont('NotoSansTC', 'normal');
            
            doc.setFontSize(18); Helpers.setHexColor(doc, '#102030');
            doc.setFont('NotoSansTC', 'bold');
            doc.text("Executive Snapshot / 總覽", M, y); y += 15;

            // --- AI 總評 ---
            doc.setFontSize(14); Helpers.setHexColor(doc, '#35637C');
            doc.setFont('NotoSansTC', 'bold');
            doc.text("AI 總評", M, y); y += 8;
            
            doc.setFontSize(10); Helpers.setHexColor(doc, '#1e293b');
            doc.setFont('NotoSansTC', 'normal');
            const commentLines = doc.splitTextToSize(result.overview_comment || '無', CONTENT_W);
            doc.text(commentLines, M, y, { lineHeightFactor: 1.5 });
            y += doc.getTextDimensions(commentLines, { lineHeightFactor: 1.5 }).h + 15;
            
            doc.setLineWidth(0.2); Helpers.setHexColor(doc, '#e2e8f0', 'draw');
            doc.line(M, y - 7, 210 - M, y - 7);

            // --- Chart ---
            doc.setFontSize(14); Helpers.setHexColor(doc, '#35637C');
            doc.setFont('NotoSansTC', 'bold');
            doc.text("分項得分率", M, y); y += 8;
            
            try {
                const chartImage = await this._getChartImage(result);
                const imgProps = doc.getImageProperties(chartImage);
                const chartHeight = CONTENT_W * (imgProps.height / imgProps.width);
                doc.addImage(chartImage, 'PNG', M, y, CONTENT_W, chartHeight);
                y += chartHeight + 15;
            } catch(e) {
                console.error("無法繪製 Chart 至 PDF:", e);
                doc.setFontSize(10);
                doc.text("無法載入圖表。", M, y);
                y += 10;
            }
            
            doc.setLineWidth(0.2); Helpers.setHexColor(doc, '#e2e8f0', 'draw');
            doc.line(M, y - 7, 210 - M, y - 7);

            // --- Strengths & Improvements ---
            if (y + 50 > 297 - 28) {
                doc.addPage();
                y = 35;
            }

            const halfW = CONTENT_W / 2 - 8;
            const startStrengthsY = y;
            
            doc.setFontSize(12);
            doc.setFont('NotoSansTC', 'bold');
            Helpers.setHexColor(doc, '#15803d');
            doc.text("主要優勢", M, startStrengthsY);

            Helpers.setHexColor(doc, '#be123c');
            doc.text("改善建議", M + halfW + 16, startStrengthsY);
            
            this.drawCategoryList(doc, result.strengths, M, startStrengthsY + 8, halfW);
            this.drawCategoryList(doc, result.improvements, M + halfW + 16, startStrengthsY + 8, halfW);
        },
        createDetailedScorePages(doc, result, parentBookmark) {
            const W = 210, H = 297, M = 22;
            let y = 35;

            const checkBreak = (needed) => {
                if (y + needed > H - 28) {
                    doc.addPage();
                    y = 35;
                }
            };
            
            for (const block of (result.breakdown || [])) {
                const blockTitle = block.id === 'report' ? '永續報告書' : '多元媒體應用';
                doc.outline.add(parentBookmark, blockTitle, { pageNumber: doc.internal.getCurrentPageInfo().pageNumber });

                for (const section of (block.sections || [])) {
                    checkBreak(20);
                    doc.setFontSize(14); doc.setFont('NotoSansTC', 'bold'); Helpers.setHexColor(doc, '#102030');
                    doc.text(section.title, M, y);
                    
                    doc.setLineWidth(0.3); Helpers.setHexColor(doc, '#35637C', 'draw');
                    doc.line(M, y + 2.5, W - M, y + 2.5);
                    y += 12;
                    
                    for (const criterion of (section.criteria || [])) {
                        const PADDING = { top: 8, bottom: 8, inner: 8, item: 8, boxMargin: 10 };
                        const LINE_HEIGHT = 1.4;
                        const RATIONALE_PADDING = 1;
                        const CRITERIA_TEXT_MAX_WIDTH = 125;
                        
                        checkBreak(40);
                        const blockStartY = y;
                        
                        const contentFinalY = (() => {
                            let currentY = blockStartY + PADDING.top;
                            currentY += doc.getTextDimensions(criterion.title).h + PADDING.inner;
                            for (const sub of (criterion.sub_criteria || [])) {
                                const titleLines = doc.splitTextToSize(`- ${sub.title}`, CRITERIA_TEXT_MAX_WIDTH);
                                currentY += doc.getTextDimensions(titleLines, { lineHeightFactor: LINE_HEIGHT }).h;
                                currentY += RATIONALE_PADDING;
                                const rationaleLines = doc.splitTextToSize(`“${sub.rationale || '無理由'}”`, CRITERIA_TEXT_MAX_WIDTH);
                                currentY += doc.getTextDimensions(rationaleLines, { lineHeightFactor: LINE_HEIGHT }).h + PADDING.item;
                            }
                            return currentY;
                        })();
                        
                        const blockHeight = (contentFinalY - blockStartY) - PADDING.item + PADDING.bottom;
                        checkBreak(blockHeight + PADDING.boxMargin);

                        doc.setLineWidth(0.4); Helpers.setHexColor(doc, '#e0e0e0', 'draw');
                        doc.roundedRect(M, y, W - 2 * M, blockHeight, 3, 3, 'S');
                        
                        let currentY = y + PADDING.top;
                        
                        // **新設計**：移除 ICON，改用左側的裝飾條
                        const titleDimensions = doc.getTextDimensions(criterion.title, { fontSize: 12 });
                        const barWidth = 2;
                        const barX = M + 5;
                        Helpers.setHexColor(doc, '#35637C', 'fill');
                        doc.rect(barX, currentY, barWidth, titleDimensions.h, 'F');

                        doc.setFontSize(12); doc.setFont('NotoSansTC', 'bold'); Helpers.setHexColor(doc, '#35637C');
                        const titleX = barX + barWidth + 4;
                        doc.text(criterion.title, titleX, currentY + 4);
                        currentY += titleDimensions.h + PADDING.inner;


                        for (const sub of (criterion.sub_criteria || [])) {
                            const scoreText = `${(sub.score || 0).toFixed(1)} / ${sub.max_score}`;
                            
                            doc.setFontSize(10);
                            const titleLines = doc.splitTextToSize(`- ${sub.title}`, CRITERIA_TEXT_MAX_WIDTH);
                            doc.setFont('NotoSansTC', 'normal'); Helpers.setHexColor(doc, '#1e293b');
                            doc.text(titleLines, M + 5, currentY, { lineHeightFactor: LINE_HEIGHT });
                            doc.text(scoreText, W - M - 5, currentY, { align: 'right' });
                            currentY += doc.getTextDimensions(titleLines, { lineHeightFactor: LINE_HEIGHT }).h;

                            currentY += RATIONALE_PADDING;

                            doc.setFontSize(9);
                            const rationaleLines = doc.splitTextToSize(`“${sub.rationale || '無理由'}”`, CRITERIA_TEXT_MAX_WIDTH);
                            Helpers.setHexColor(doc, '#475569');
                            doc.text(rationaleLines, M + 8, currentY, { maxWidth: CRITERIA_TEXT_MAX_WIDTH, lineHeightFactor: LINE_HEIGHT });
                            currentY += doc.getTextDimensions(rationaleLines, { lineHeightFactor: LINE_HEIGHT }).h + PADDING.item;
                        }
                        y += blockHeight + PADDING.boxMargin;
                    }
                }
            }
        },
    };

    // --- 應用程式邏輯模組 ---
    const Logic = {
        state: null,
        
        init(appState) {
            this.state = appState;
        },

        handleFiles(files) {
            if (this.state.isMockMode) { this.state.files = []; this.state.websites = []; }
            for (const file of files) {
                if (file.type === 'application/pdf') {
                    const fileObject = { file: file, id: 'file-' + Date.now() + Math.random(), name: file.name, size: file.size };
                    this.state.files.push(fileObject);
                    const companyName = file.name.replace(/\.pdf$/i, '');
                    if (!this.state.websites.some(w => w.company === companyName)) {
                        this.addWebsite(companyName, 'https://mock.example.com');
                    }
                }
            }
            UI.renderFileList();
            UI.renderWebsiteList();
        },

        addWebsite(companyName = '', url = '') {
            const websiteObject = { id: 'site-' + Date.now() + Math.random(), company: companyName, url: url };
            this.state.websites.push(websiteObject);
            UI.renderWebsiteList();
        },

        async startProcessing() {
            if (this.state.isProcessing) return;
            if (this.state.isMockMode) {
                UI.changeStep(3);
                document.getElementById('progress-text').textContent = '正在載入預覽資料...';
                setTimeout(() => {
                    this.state.results = MOCK_RESULTS;
                    UI.renderResults();
                    UI.changeStep(4);
                }, 1500);
                return;
            }
            // ... (與後端通訊的邏輯)
            const matchedData = [];
            let hasError = false;
            this.state.websites.forEach(site => {
                const correspondingFile = this.state.files.find(f => f.name.replace(/\.pdf$/i, '') === site.company);
                if (!site.company.trim()) { alert(`錯誤：有一個公司名稱欄位是空的。`); hasError = true; return; }
                if (correspondingFile) {
                    matchedData.push({ file: correspondingFile.file, company: site.company, url: site.url || 'N/A' });
                }
            });
            if (hasError) return;
            if (matchedData.length === 0) { alert('請至少上傳一份 PDF，並確保其檔名（不含 .pdf）與公司名稱欄位完全相符。'); return; }
            this.state.isProcessing = true;
            UI.changeStep(3);
            try {
                const simpleFormData = new FormData();
                matchedData.forEach(item => {
                    simpleFormData.append('files', item.file, item.file.name);
                    simpleFormData.append('company_names', item.company);
                    simpleFormData.append('website_urls', item.url);
                });
                document.getElementById('progress-text').textContent = `正在準備上傳 ${matchedData.length} 個檔案...`;
                const response = await fetch(`${Config.API_BASE_URL}/scoring/batch`, { method: 'POST', body: simpleFormData });
                document.getElementById('progress-text').textContent = '後端處理中，請稍候...';
                if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.detail || `伺服器錯誤: ${response.status}`); }
                this.state.results = await response.json();
                UI.renderResults();
                UI.changeStep(4);
            } catch (error) {
                console.error('處理失敗:', error);
                alert(`發生錯誤：${error.message}`);
                UI.changeStep(2);
            } finally {
                this.state.isProcessing = false;
            }
        },
        
        resetApp() {
            const wasInMockMode = this.state.isMockMode;
            Object.assign(this.state, { currentStep: 1, files: [], websites: [], results: [], isProcessing: false });
            UI.renderFileList(); 
            UI.renderWebsiteList();
            if (wasInMockMode) { this.exitPreviewMode(); } 
            else { UI.changeStep(1); }
        },
        
        startPreviewMode() {
            this.state.isMockMode = true;
            document.body.classList.add('preview-mode');
            UI.elements.previewBanner.style.display = 'block';
            UI.changeStep(2);
        },
        
        exitPreviewMode() {
            this.state.isMockMode = false;
            document.body.classList.remove('preview-mode');
            UI.elements.previewBanner.style.display = 'none';
            this.resetApp();
        },

        async testBackendConnection() {
            const btn = document.getElementById('test-connection-btn'); 
            const statusDiv = document.getElementById('connection-status');
            const originalText = btn.querySelector('#test-text').textContent;
            btn.disabled = true; 
            btn.querySelector('#test-text').textContent = "測試中...";
            statusDiv.classList.remove('hidden', 'text-green-600', 'text-red-600'); 
            statusDiv.textContent = '';
            try {
                const response = await fetch(`${Config.API_BASE_URL}/health`);
                if (response.ok) {
                    const data = await response.json(); 
                    statusDiv.textContent = `✅ 連接成功: ${data.message}`;
                    statusDiv.classList.add('text-green-600');
                } else { 
                    throw new Error(`伺服器回應錯誤: ${response.status}`); 
                }
            } catch (error) {
                statusDiv.textContent = `❌ 連接失敗: ${error.message}. 請確認後端伺服器是否已啟動。`;
                statusDiv.classList.add('text-red-600');
            } finally {
                btn.disabled = false; 
                btn.querySelector('#test-text').textContent = originalText;
                statusDiv.classList.remove('hidden');
            }
        }
    };

    // --- 主應用程式控制器 ---
    const App = {
        state: {
            currentStep: 1,
            files: [],
            websites: [],
            results: [],
            fontData: { normal: null, bold: null }, // 用於儲存 Base64 字型
            isProcessing: false,
            isMockMode: false,
        },

        async init() {
            // 初始化各模組
            UI.init(this.state);
            PDFGenerator.init(this.state);
            Logic.init(this.state);
            
            // 綁定事件監聽器
            this.bindEvents();
            
            // 更新初始 UI
            UI.updateStepUI();

            // 非同步載入字型
            this.loadInitialFonts();
        },

        async loadInitialFonts() {
            if (UI.elements.fontStatus) {
                UI.elements.fontStatus.innerHTML = `
                    <svg class="animate-spin h-5 w-5 text-primary-accent" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    正在載入 PDF 所需字型...`;
            }
                
            const normalFont = await Helpers.loadFontAsBase64(Config.FONT_PATHS.normal);

            // **修正**：無論 Bold 字體是否存在，都將 normal 字體資料赋給 bold
            this.state.fontData.normal = normalFont && !normalFont.error ? normalFont : null;
            this.state.fontData.bold = this.state.fontData.normal; // Bold 使用與 Normal 相同的字體資料

            if (UI.elements.fontStatus) {
                if (this.state.fontData.normal) {
                    UI.elements.fontStatus.innerHTML = `✅ PDF 所需字型已成功載入。`;
                    UI.elements.fontStatus.className = 'text-sm text-green-600 flex items-center gap-2';
                } else {
                    const errorMsg = (normalFont && normalFont.error) || "未知錯誤";
                    UI.elements.fontStatus.innerHTML = `
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        <span>警告：PDF 字型載入失敗，匯出功能可能無法使用。<br><small class="text-gray-500">${errorMsg}</small></span>`;
                    UI.elements.fontStatus.className = 'text-sm text-red-600 flex items-start gap-2';
                }
            }
        },

        bindEvents() {
            document.getElementById('next-step1').addEventListener('click', () => UI.changeStep(2));
            document.getElementById('prev-step2').addEventListener('click', () => UI.changeStep(1));
            document.getElementById('next-step2').addEventListener('click', () => Logic.startProcessing());
            document.getElementById('back-to-start').addEventListener('click', () => Logic.resetApp());
            
            const dropArea = document.getElementById('drop-area');
            dropArea.addEventListener('click', () => document.getElementById('file-input').click());
            document.getElementById('file-input').addEventListener('change', (e) => Logic.handleFiles(e.target.files));
            
            const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
            });
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
            });
            dropArea.addEventListener('drop', (e) => Logic.handleFiles(e.dataTransfer.files), false);
            
            document.getElementById('add-website-btn').addEventListener('click', () => Logic.addWebsite('', ''));
            document.getElementById('start-preview-btn').addEventListener('click', () => Logic.startPreviewMode());
            document.getElementById('exit-preview-btn').addEventListener('click', () => Logic.exitPreviewMode());
            document.getElementById('test-connection-btn').addEventListener('click', () => Logic.testBackendConnection());
        }
    };

    // --- 範例資料 ---
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
                  {"title": "架構", "score": 8.0, "max_score": 10.0, "sub_criteria": [ {"title": "是否清楚整理並呈現本年度的亮點作為報告書的總結", "max_score": 3.0, "score": 3.0, "rationale": "報告書前段有 Highlight 整理。"}, {"title": "完整的索引設計(包括GRI, SASB及其他重要規範等)", "max_score": 3.0, "score": 3.0, "rationale": "附錄提供完整 GRI/SASB 索引。"}, {"title": "報告書附有清楚的連結，使讀者可透過網頁的說明獲得更細節的資訊", "max_score": 2.0, "score": 1.0, "rationale": "部分連結可點擊，但並非全部。"}, {"title": "架構呈現完整易于查閱", "max_score": 2.0, "score": 1.0, "rationale": "目錄清晰，但缺乏互動式導覽。"} ]}
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

    // --- 應用程式啟動 ---
    App.init();
});

