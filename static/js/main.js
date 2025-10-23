document.addEventListener('DOMContentLoaded', () => {

    // --- 應用程式設定 ---
    const Config = {
        API_BASE_URL: 'http://127.0.0.1:8000',
        FONT_PATHS: {
            normal: '/static/Font/NotoSansTC-Regular.ttf',
        },
        COVER_IMAGE_BASE64: "", // 可選：貼上封面圖片的 Base64
    };

    // --- 輔助工具模組 ---
    const Helpers = {
        loadFontAsBase64: async (url) => {
            try {
                // 優先使用嵌入的 Base64 (如果 window.__FONT_BASE64__ 存在且有效)
                if (window.__FONT_BASE64__ && window.__FONT_BASE64__.length > 100) {
                    try {
                        const base64Data = window.__FONT_BASE64__.split(',')[1] || window.__FONT_BASE64__;
                        atob(base64Data); // 驗證 Base64
                        console.log("Using embedded font data."); // 確認使用嵌入字型
                        return base64Data;
                    } catch (e) {
                        console.error("嵌入的 Base64 字型無效:", e);
                        // 若嵌入的無效，則繼續嘗試從伺服器抓取
                    }
                }
                console.log(`Fetching font from URL: ${url}`); // 確認從 URL 抓取
                // 從伺服器抓取
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
                return { error: `無法從 ${url} 載入字型。請檢查後端 CORS 設定與檔案路徑，或確認已在 index.html 中嵌入有效的 Base64 字串。錯誤詳情: ${error.message}` };
            }
        },
        setHexColor: (doc, hex, type = 'text') => {
            if (!hex || typeof hex !== 'string' || hex.length < 4 || !hex.startsWith('#')) { // 檢查 # 開頭
                console.warn(`Invalid hex color format: ${hex}. Using default.`);
                // 設定預設顏色避免錯誤
                if (type === 'text') doc.setTextColor(0, 0, 0); // Black
                else if (type === 'draw') doc.setDrawColor(0, 0, 0); // Black
                else if (type === 'fill') doc.setFillColor(200, 200, 200); // Gray
                return;
            }
            try {
                const r = parseInt(hex.substring(1, 3), 16);
                const g = parseInt(hex.substring(3, 5), 16);
                const b = parseInt(hex.substring(5, 7), 16);
                if (isNaN(r) || isNaN(g) || isNaN(b)) throw new Error("Invalid hex component");

                if (type === 'text') doc.setTextColor(r, g, b);
                else if (type === 'draw') doc.setDrawColor(r, g, b);
                else if (type === 'fill') doc.setFillColor(r, g, b);
            } catch (e) {
                 console.error(`Error setting hex color ${hex}:`, e);
                 // 設定預設顏色避免錯誤
                if (type === 'text') doc.setTextColor(0, 0, 0);
                else if (type === 'draw') doc.setDrawColor(0, 0, 0);
                else if (type === 'fill') doc.setFillColor(200, 200, 200);
            }
        },
        sanitizeFileName: (name) => (name || 'report').replace(/[\\/:*?"<>|\n\r]+/g, '_').slice(0, 128),
        // v6.2 移除 addVerticalGradient
    };

    // --- UI 介面管理模組 ---
    const UI = {
        elements: {},
        init(appState) {
            this.state = appState;
            this.elements = {
                steps: [document.getElementById('step1'), document.getElementById('step2'), document.getElementById('step3'), document.getElementById('step4')],
                stepWrappers: [document.getElementById('step-wrapper-1'), document.getElementById('step-wrapper-2'), document.getElementById('step-wrapper-3'), document.getElementById('step-wrapper-4')],
                stepLines: [document.getElementById('step-line-1'), document.getElementById('step-line-2'), document.getElementById('step-line-3')],
                contents: [document.getElementById('step1-content'), document.getElementById('step2-content'), document.getElementById('step3-content'), document.getElementById('step4-content')],
                fileList: document.getElementById('file-list'),
                websiteList: document.getElementById('website-list'),
                resultsContainer: document.getElementById('results-container'),
                previewBanner: document.getElementById('preview-banner'),
                fontStatusContainer: document.getElementById('font-status-container'), // Corrected ID usage
            };
        },
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
        updateStepUI() {
            this.elements.contents.forEach((content, index) => {
                 if (content) content.classList.toggle('hidden', (index + 1) !== this.state.currentStep);
            });
            this.elements.stepWrappers.forEach((wrapper, index) => {
                 if (!wrapper) return;
                const stepNum = index + 1;
                const indicator = wrapper.firstElementChild; // Get the inner div
                wrapper.classList.remove('step-active', 'step-completed');
                 if (!indicator) return; // Guard against missing indicator
                indicator.textContent = stepNum;
                indicator.innerHTML = stepNum; // Reset to number first

                if (stepNum < this.state.currentStep) {
                    wrapper.classList.add('step-completed');
                    indicator.innerHTML = `&#10003;`; // Checkmark
                } else if (stepNum === this.state.currentStep) {
                    wrapper.classList.add('step-active');
                    // indicator remains the number
                }
            });
            this.elements.stepLines.forEach((line, index) => {
                if (!line) return;
                line.classList.remove('step-line-active', 'step-line-completed');
                if (index + 1 < this.state.currentStep) {
                    line.classList.add('step-line-completed');
                } else if (index + 1 === this.state.currentStep - 1) {
                    // This logic seems correct for the line *before* the active step
                    line.classList.add('step-line-active');
                }
            });
        },
        renderFileList() {
             if (!this.elements.fileList) return;
            this.elements.fileList.innerHTML = this.state.files.map(f => this.templates.fileItem(f)).join('');
            this.elements.fileList.querySelectorAll('.remove-file-btn').forEach(btn => {
                btn.removeEventListener('click', this.handleRemoveFile);
                btn.addEventListener('click', this.handleRemoveFile.bind(this));
            });
        },
         handleRemoveFile(e) {
            const fileId = e.currentTarget.dataset.id;
            const fileToRemove = this.state.files.find(file => file.id === fileId);
            if (!fileToRemove) return; // Exit if file not found

            this.state.files = this.state.files.filter(file => file.id !== fileId);
             // After removing a file, also remove the corresponding website entry if it exists
             const companyName = fileToRemove.name.replace(/\.pdf$/i, '').trim();
             this.state.websites = this.state.websites.filter(site => site.company !== companyName);

            this.renderFileList();
            this.renderWebsiteList(); // Re-render website list as well
         },
        renderWebsiteList() {
             if (!this.elements.websiteList) return;
            this.elements.websiteList.innerHTML = this.state.websites.map(w => this.templates.websiteItem(w)).join('');
            this.elements.websiteList.querySelectorAll('.remove-website-btn').forEach(btn => {
                btn.removeEventListener('click', this.handleRemoveWebsite);
                btn.addEventListener('click', this.handleRemoveWebsite.bind(this));
            });
            this.elements.websiteList.querySelectorAll('input').forEach(input => {
                input.removeEventListener('change', this.handleWebsiteInputChange);
                input.addEventListener('change', this.handleWebsiteInputChange.bind(this));
            });
        },
        handleRemoveWebsite(e) {
            const siteId = e.currentTarget.dataset.id;
             // Also remove the corresponding file if it exists
             const siteToRemove = this.state.websites.find(site => site.id === siteId);
             if(siteToRemove) {
                 this.state.files = this.state.files.filter(file => !file.name.startsWith(siteToRemove.company));
             }

            this.state.websites = this.state.websites.filter(site => site.id !== siteId);
            this.renderWebsiteList();
            this.renderFileList(); // Re-render file list
        },
        handleWebsiteInputChange(e) {
            const siteId = e.currentTarget.dataset.id;
            const field = e.currentTarget.dataset.field;
            const site = this.state.websites.find(s => s.id === siteId);
            if (site) {
                site[field] = e.target.value;
                 // Maybe add validation or feedback here if needed
            }
        },
        renderResults() {
             if (!this.elements.resultsContainer) return;
            this.elements.resultsContainer.innerHTML = this.state.results.map(r => this.templates.resultCard(r)).join('');
            this.state.results.forEach(r => {
                const companyId = this.templates.sanitizeId(r.company); // Ensure ID is sanitized correctly
                const exportBtn = document.querySelector(`.export-pdf-btn[data-company-id="${companyId}"]`);
                if (exportBtn) {
                     // Use a unique property to store the bound handler if needed, or manage listeners differently
                     const boundHandler = this.handleExportPdf.bind(this, r, exportBtn);
                     exportBtn.removeEventListener('click', exportBtn._boundClickHandler); // Remove previous if stored
                     exportBtn.addEventListener('click', boundHandler);
                     exportBtn._boundClickHandler = boundHandler; // Store for potential removal
                }
                const canvasId = `chart-${companyId}`;
                const ctx = document.getElementById(canvasId);
                if (ctx) {
                    const existingChart = Chart.getChart(ctx); // Use Chart.getChart
                    if (existingChart) {
                        existingChart.destroy();
                    }
                    this.createChart(ctx, r);
                } else {
                     console.warn(`Canvas element with ID ${canvasId} not found.`);
                 }

                const toggleBtn = document.getElementById(`toggle-details-${companyId}`);
                const detailsPanel = document.getElementById(`details-${companyId}`);
                if (toggleBtn && detailsPanel) {
                      const boundToggleHandler = this.handleToggleDetails.bind(this, detailsPanel, toggleBtn);
                      toggleBtn.removeEventListener('click', toggleBtn._boundClickHandler); // Use stored handler
                      toggleBtn.addEventListener('click', boundToggleHandler);
                      toggleBtn._boundClickHandler = boundToggleHandler; // Store new handler
                     const isHidden = detailsPanel.classList.contains('hidden');
                     toggleBtn.textContent = isHidden ? '查看詳細評分' : '隱藏詳細評分';
                }

                if (detailsPanel && !detailsPanel.classList.contains('hidden')) { // Bind only if visible initially? Or always? Always is safer.
                    this.bindDetailViewEvents(detailsPanel);
                }
            });
        },
         handleExportPdf(result, buttonElement, event) {
             // Add check for font data before generating
             if (!this.state.fontData.normal && !this.state.isMockMode) {
                 alert('錯誤：PDF 所需字型未載入，無法匯出。');
                 return;
             }
             PDFGenerator.generate(result, buttonElement);
         },
         handleToggleDetails(detailsPanel, toggleBtn, event) {
             const isHidden = detailsPanel.classList.contains('hidden');
             detailsPanel.classList.toggle('hidden', !isHidden);
             toggleBtn.textContent = isHidden ? '隱藏詳細評分' : '查看詳細評分';
             // If showing details, ensure events are bound
             if (!isHidden) {
                 this.bindDetailViewEvents(detailsPanel);
             } else {
                 // Optional: Disconnect observer when hidden to save resources
                 const observer = detailsPanel.dataset.scrollObserver;
                 // Check if observer exists and is an IntersectionObserver instance
                 if (observer && observer instanceof IntersectionObserver && typeof observer.disconnect === 'function') {
                     observer.disconnect();
                     delete detailsPanel.dataset.scrollObserver; // Clean up dataset attribute
                     // Also remove the eventsBound flag? Or keep it? Let's remove it.
                     delete detailsPanel.dataset.eventsBound;
                 }
             }
         },
        bindDetailViewEvents(container) {
            // Check if events are already bound to prevent duplicates
            if (container.dataset.eventsBound === 'true') {
                console.log("Detail view events already bound for:", container.id);
                // Re-initialize observer if needed, especially if container was hidden/shown
                 const observer = container.dataset.scrollObserver;
                 if (observer && observer instanceof IntersectionObserver) {
                    // Assuming observer is stored correctly
                     console.log("Re-observing sections...");
                     const contentSections = container.querySelectorAll('.detail-content-section');
                     contentSections.forEach(section => observer.observe(section));
                 } else {
                     console.log("No observer found or observer invalid, re-binding scrollspy.");
                     // Force re-binding if observer is missing
                      delete container.dataset.eventsBound; // Allow re-binding below
                 }
                // return; // Decide if we should exit or always re-bind observer
            }

            const navLinks = container.querySelectorAll('.detail-nav-link');
            const contentSections = container.querySelectorAll('.detail-content-section');
            const mainContentArea = container.querySelector('main');

            if (!mainContentArea || navLinks.length === 0 || contentSections.length === 0) {
                 console.warn("Detail view elements missing, cannot bind events.");
                 return; // Exit if essential elements not found
             }


            // --- Navigation Link Click Handling ---
            const handleNavLinkClick = (e) => {
                e.preventDefault();
                const targetId = e.currentTarget.getAttribute('href')?.substring(1);
                 if (!targetId) return;
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    const details = targetElement.querySelector('details');
                    if (details && !details.open) {
                        details.open = true;
                    }
                    setTimeout(() => {
                        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 50);
                }
            };

            navLinks.forEach(link => {
                link.removeEventListener('click', handleNavLinkClick); // Clean up first
                link.addEventListener('click', handleNavLinkClick);
            });


            // --- Scrollspy Logic ---
            const observerCallback = (entries) => {
                 let mostVisibleEntry = null;
                 let maxRatio = 0;

                 entries.forEach(entry => {
                     if (entry.isIntersecting) {
                         if (entry.intersectionRatio > maxRatio) {
                             maxRatio = entry.intersectionRatio;
                             mostVisibleEntry = entry;
                         }
                         else if (entry.intersectionRatio === maxRatio) {
                             if (!mostVisibleEntry || entry.boundingClientRect.top < mostVisibleEntry.boundingClientRect.top) {
                                 mostVisibleEntry = entry;
                             }
                         }
                     }
                 });

                 if (mostVisibleEntry) {
                     const id = mostVisibleEntry.target.getAttribute('id');
                     navLinks.forEach(link => {
                         const isActive = link.getAttribute('href') === `#${id}`;
                         link.classList.toggle('font-semibold', isActive);
                         link.style.backgroundColor = isActive ? '#E6EEF3' : '';
                         link.style.color = isActive ? '#203F58' : ''; // Reset to default/inherit if not active
                         if (isActive) {
                            link.setAttribute('aria-current', 'location');
                         } else {
                            link.removeAttribute('aria-current');
                         }
                     });
                 }
             };


            const observerOptions = {
                root: mainContentArea,
                rootMargin: "0px 0px -70% 0px", // Observe changes in the top 30%
                threshold: 0.1 // Trigger early
            };
            // Ensure any previous observer is disconnected before creating a new one
            const existingObserver = container.dataset.scrollObserver;
             if (existingObserver && existingObserver instanceof IntersectionObserver) {
                 existingObserver.disconnect();
             }

            const observer = new IntersectionObserver(observerCallback, observerOptions);
            contentSections.forEach(section => observer.observe(section));

             // Store the new observer instance using a direct property or weakmap if preferred
             // Using dataset is simpler here
             container.dataset.scrollObserver = observer; // Storing the instance itself might not work reliably across sessions/clones. Store a flag instead?
             // Let's store a flag and manage the observer instance internally if needed, or rely on re-binding.
             // For simplicity, let's assume storing the observer might work, but be cautious. A better way might involve managing observers centrally.


            container.dataset.eventsBound = 'true'; // Mark as bound
            console.log("Detail view events bound for:", container.id);
        },
        _getChartConfig(result) {
            // ... (v6.2 logic - seems okay, using PDFGenerator colors) ...
            const reportBreakdown = result.breakdown?.find(b => b.id === 'report');
            const mediaBreakdown = result.breakdown?.find(b => b.id === 'media');
            const labels = [], scores = [], maxScores = [];
            if (reportBreakdown?.sections) {
                reportBreakdown.sections.forEach(sec => { labels.push(sec.title); scores.push(sec.score ?? 0); maxScores.push(sec.max_score ?? 0); });
            }
            if (mediaBreakdown?.sections) {
                 mediaBreakdown.sections.forEach(sec => {
                     labels.push(sec.title);
                     scores.push(sec.score ?? 0); // Directly use section score
                     maxScores.push(sec.max_score ?? 0); // Use section max_score
                 });
            }
            const percentages = scores.map((score, i) => maxScores[i] > 0 ? (score / maxScores[i]) * 100 : 0);
            const chartTextColor = '#4b5563'; // Tailwind gray-600

            const colors = [];
            if (reportBreakdown?.sections) {
                reportBreakdown.sections.forEach(() => colors.push(PDFGenerator.COLORS.primary));
            }
            if (mediaBreakdown?.sections) {
                mediaBreakdown.sections.forEach(() => colors.push(PDFGenerator.COLORS.media));
            }

            return {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '得分率 (%)',
                        data: percentages,
                        backgroundColor: colors.map(c => c + '99'), // Add alpha transparency
                        borderColor: colors,
                        borderWidth: 1,
                        borderRadius: 4,
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: false, // Set to false for PDF generation context
                    maintainAspectRatio: true, // Maintain aspect ratio based on height/width
                    animation: false, // Disable animation for PDF context
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                color: chartTextColor,
                                callback: (v) => v + "%",
                                font: { size: 12 } // Adjust font size for PDF
                            },
                            grid: { color: 'rgba(209, 213, 219, 0.4)' } // Tailwind gray-300 with alpha
                        },
                        y: {
                            ticks: {
                                color: chartTextColor,
                                font: { size: 14 }, // Adjust font size for PDF
                                autoSkip: false // Ensure all labels are shown if possible
                            },
                            grid: { display: false }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false } // Disable tooltips for PDF context
                    },
                    layout: {
                        padding: { left: 10, right: 20, top: 10, bottom: 10 } // Adjust padding
                    }
                }
            };
        },
        createChart(ctx, result) {
            // ... (v6.2 logic - seems okay) ...
             const config = this._getChartConfig(result);
            // --- Adjustments for Live HTML Chart ---
            config.options.responsive = true; // Enable responsiveness for live chart
            config.options.maintainAspectRatio = false; // Allow chart to fill container
            config.options.animation = true; // Enable animation for live chart
            config.options.plugins.tooltip.enabled = true; // Enable tooltips for live chart
            config.options.plugins.tooltip.callbacks = {
                // Correctly access label and parsed value
                 label: (tooltipItem) => {
                     let label = tooltipItem.dataset.label || '';
                     if (label) {
                         label += ': ';
                     }
                     if (tooltipItem.parsed.x !== null) {
                         label += `${tooltipItem.parsed.x.toFixed(1)}%`;
                     }
                     return label;
                 },
                 // Optional: Customize title if needed
                 title: (tooltipItems) => {
                     return tooltipItems[0]?.label || ''; // Use the y-axis label as title
                 }
            };

            config.options.scales.y.ticks.font.size = 10; // Smaller font for Y-axis on live chart
             config.options.scales.x.ticks.font.size = 10; // Smaller font for X-axis on live chart
             config.options.scales.y.ticks.autoSkip = true; // Allow skipping labels if too crowded

            // Adjust text color based on preview mode (using CSS variables)
            const isPreview = document.body.classList.contains('preview-mode');
            // Ensure fallback colors if CSS variables are not defined
            const secondaryTextColor = isPreview
                ? (getComputedStyle(document.documentElement).getPropertyValue('--text-secondary')?.trim() || '#64748B')
                : (getComputedStyle(document.documentElement).getPropertyValue('--text-secondary')?.trim() || '#64748B');


            config.options.scales.x.ticks.color = secondaryTextColor;
            config.options.scales.y.ticks.color = secondaryTextColor;

            try {
                new Chart(ctx, config);
            } catch (e) {
                console.error("Error creating live chart:", e);
                // Optionally display an error message on the canvas
                const context = ctx.getContext('2d');
                if (context) {
                    context.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                    context.fillStyle = 'red';
                    context.font = '12px Arial';
                    context.fillText('無法載入圖表', 10, 20);
                }
            }
        },
        templates: {
            fileItem(file) {
                // ... (v6.2 - improved truncation) ...
                return `<div class="flex items-center justify-between bg-gray-50 p-3 rounded-md border" style="border-color: var(--border-color);">
                            <div class="flex items-center gap-3 overflow-hidden">
                                <svg class="w-6 h-6 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 2a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V8.344a1.996 1.996 0 00-.6-.344l-5-2.5a2 2 0 00-1.932 0l-5 2.5A1.996 1.996 0 004 8.344V4a2 2 0 012-2h4a2 2 0 110 4H6V4a2 2 0 00-2-2z" clip-rule="evenodd" /><path d="M11 11.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" /></svg>
                                <div class="truncate">
                                    <p class="text-sm font-medium text-text-primary truncate" title="${file.name}">${file.name}</p>
                                    <p class="text-xs text-text-secondary">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                            </div>
                            <button data-id="${file.id}" class="remove-file-btn p-1.5 rounded-full hover:bg-gray-200 text-gray-500 hover:text-gray-700 flex-shrink-0 ml-2">
                                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
                            </button>
                        </div>`;
            },
            websiteItem(site) {
                // ... (v6.2 - using CSS class for input) ...
                 return `<div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                            <div class="md:col-span-5">
                                <input type="text" data-id="${site.id}" data-field="company" value="${site.company}" placeholder="企業名稱 (須與檔名相同)" class="input-field w-full">
                            </div>
                            <div class="md:col-span-6">
                                <input type="text" data-id="${site.id}" data-field="url" value="${site.url}" placeholder="企業永續網站 URL (選填)" class="input-field w-full">
                            </div>
                            <div class="md:col-span-1 text-right">
                                <button data-id="${site.id}" class="remove-website-btn p-1.5 rounded-full hover:bg-gray-200 text-gray-500 hover:text-gray-700">
                                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clip-rule="evenodd" /></svg>
                                </button>
                            </div>
                        </div>`;
            },
            resultCard(result) {
                // Use standard HTML comments or remove them
                 const companyId = this.sanitizeId(result.company);
                const level = this.getLevelAndColor(result.totals?.final, true);
                const isPreview = result.company.includes("(預覽)");
                const mainTitle = result.company.replace(" (預覽)", "");
                const scoreValue = (result.totals?.final ?? '-').toFixed(1);

                return `<div class="card p-6 md:p-8 shadow-sm border border-gray-200 rounded-lg">
                    <!-- Card Header -->
                    <div class="flex flex-wrap justify-between items-start gap-4 mb-6">
                        <div class="flex items-center gap-3">
                            <h3 class="text-2xl font-bold text-text-primary">${mainTitle}</h3>
                            ${isPreview ? '<span class="badge-preview">預覽</span>' : ''}
                        </div>
                        <button data-company-id="${companyId}" class="export-pdf-btn btn btn-primary">
                            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                            匯出 PDF 報告
                        </button>
                    </div>
                     <p class="text-md font-normal text-text-secondary -mt-6 mb-8">AI 評分分析報告</p>

                    <!-- Main Content Grid -->
                    <div class="grid grid-cols-1 lg:grid-cols-5 gap-8">
                        <!-- Left Column -->
                        <div class="lg:col-span-3 space-y-8">
                            <div>
                                <h4 class="text-base font-semibold text-text-primary">關鍵摘要</h4>
                                <hr class="mt-3 mb-4 hr-style" />
                                <div class="space-y-4 max-w-prose leading-relaxed text-text-secondary">
                                    <p><strong>AI 總評:</strong> ${result.overview_comment || '無'}</p>
                                    <p class="text-xs text-gray-500">評分日期: ${new Date().toLocaleDateString('zh-TW')} · Gemini v2.5</p>
                                </div>
                                <div class="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
                                    <div class="card p-4 shadow-none border border-gray-200 rounded-lg">
                                        <h5 class="font-semibold text-strengths mb-2 flex items-center"><svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 16 16"><path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/></svg>主要優勢</h5>
                                        ${this.renderStrengthsImprovements(result.strengths)}
                                    </div>
                                    <div class="card p-4 shadow-none border border-gray-200 rounded-lg">
                                        <h5 class="font-semibold text-improvements mb-2 flex items-center"><svg class="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 16 16"><path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/></svg>改善建議</h5>
                                        ${this.renderStrengthsImprovements(result.improvements)}
                                    </div>
                                </div>
                            </div>
                            <div class="border-t pt-8 hr-style">
                               <button id="toggle-details-${companyId}" class="btn btn-secondary w-full">查看詳細評分</button>
                            </div>
                        </div>

                        <!-- Right Column -->
                        <div class="lg:col-span-2 space-y-8">
                           <div class="w-full flex justify-center lg:justify-end">
                                <div class="text-center p-4 rounded-lg cursor-default w-full shadow-inner" style="background-color: ${level.bgColor}; border: 1px solid ${level.borderColor};">
                                    <p class="text-4xl font-bold" style="color: ${level.textColor};">${scoreValue}</p>
                                    <p class="text-sm font-semibold" style="color: ${level.textColor};">${level.text}</p>
                                </div>
                           </div>
                           <div>
                                <h4 class="text-base font-semibold text-text-primary">分項得分率</h4>
                                <hr class="mt-3 mb-4 hr-style" />
                                <div class="chart-container relative h-64 md:h-80 lg:h-96">
                                    <canvas id="chart-${companyId}"></canvas>
                                </div>
                           </div>
                        </div>
                    </div>

                    <div id="details-${companyId}" class="mt-8 border-t pt-8 hidden hr-style">
                        ${this.renderDetailedView(result, companyId)}
                    </div>
                </div>`;
            },
             renderStrengthsImprovements(data) {
                 // ... (v6.2 logic - seems okay) ...
                  // Use "總覽" key if available, otherwise flatten all lists from other keys
                 const summaryItems = data?.總覽 && Array.isArray(data.總覽) && data.總覽.length > 0 ? data.總覽 : null;
                 let itemsToList = [];

                 if (summaryItems) {
                    itemsToList = summaryItems;
                 } else {
                    // Flatten items from all keys except '總覽' if summary is missing
                     itemsToList = Object.entries(data || {})
                                    .filter(([key]) => key !== '總覽')
                                    .flatMap(([, value]) => Array.isArray(value) ? value : []);
                 }


                 // Filter out empty or "無" items
                 const validItems = itemsToList.filter(item => item && typeof item === 'string' && !item.includes("無具體") && item.trim() !== "" && item.trim() !== "無");


                 if (validItems.length === 0) {
                     return '<ul class="list-disc pl-5 mt-1 text-sm text-text-secondary space-y-1"><li>無</li></ul>';
                 }

                 let html = '<ul class="list-disc pl-5 mt-1 text-sm text-text-secondary space-y-1">';
                 // Limit number of items shown? For now, show all valid ones.
                 html += validItems.map(item => `<li>${item}</li>`).join('');
                 html += '</ul>';
                 return html;
             },
            renderDetailedView(result, companyId) {
                // *** CORRECTION: Remove incorrect comments ***
                 const renderSummaryCard = (section) => {
                    const strengths = (result.strengths && result.strengths[section.title]) || [];
                    const improvements = (result.improvements && result.improvements[section.title]) || [];

                     const strengthItems = strengths.filter(item => item && typeof item === 'string' && !item.includes("無具體") && item.trim() !== "" && item.trim() !== "無");
                     const improvementItems = improvements.filter(item => item && typeof item === 'string' && !item.includes("無具體") && item.trim() !== "" && item.trim() !== "無");

                    // Use standard HTML comments or remove them
                    return `
                        <div class="bg-[#F3F4F6] border border-[#E9E4DF] shadow-inner rounded-xl mt-6 p-4">
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
                                <div>
                                    <h4 class="font-semibold text-[#35637C] mb-2 flex items-center"><svg class="w-4 h-4 mr-1.5 flex-shrink-0" fill="currentColor" viewBox="0 0 16 16"><path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/></svg>主要優勢</h4>
                                    <ul class="list-disc pl-5 space-y-1 text-sm text-[#1E293B]">
                                        ${strengthItems.length > 0 ? strengthItems.map(item => `<li>${item}</li>`).join('') : '<li class="text-gray-500">無</li>'}
                                    </ul>
                                </div>
                                <div>
                                    <h4 class="font-semibold text-[#F69268] mb-2 flex items-center"><svg class="w-4 h-4 mr-1.5 flex-shrink-0" fill="currentColor" viewBox="0 0 16 16"><path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/></svg>改善建議</h4>
                                    <ul class="list-disc pl-5 space-y-1 text-sm text-[#1E293B]">
                                         ${improvementItems.length > 0 ? improvementItems.map(item => `<li>${item}</li>`).join('') : '<li class="text-gray-500">無</li>'}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    `;
                };

                 // Calculate total number of sections for the scrollspy reference
                const totalSections = (result.breakdown || []).reduce((acc, block) => acc + (block.sections || []).length, 0);

                 // Use standard HTML comments or remove them
                return `
                <style>
                    details > summary { list-style: none; cursor: pointer; }
                    details > summary::-webkit-details-marker { display: none; }
                    details[open] > summary .details-arrow { transform: rotate(180deg); }
                    .scroll-mt-20 { scroll-margin-top: 5rem; /* Adjust if header height changes */ }
                    .detail-nav-link[aria-current="location"] {
                         background-color: #E6EEF3;
                         color: #203F58 !important;
                         font-weight: 600;
                     }
                    .smooth-scroll-container { scroll-behavior: smooth; }
                </style>
                <div class="px-0 md:px-6 py-8">
                    <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
                        <!-- Sidebar Navigation -->
                        <aside class="lg:col-span-1 lg:sticky top-10 self-start" style="max-height: calc(100vh - 5rem); overflow-y: auto;">
                            <div class="bg-white border border-[#E9E4DF] rounded-lg p-3">
                                <h5 class="font-semibold mb-3 text-[#1E293B] px-2 text-base">評分章節 (${totalSections})</h5>
                                <nav aria-label="詳細評分章節">
                                    <ul class="space-y-1">
                                        ${(result.breakdown || []).flatMap(block => block.sections).map(section => `
                                            <li>
                                                <a href="#section-${this.sanitizeId(section.title)}-${companyId}"
                                                   class="detail-nav-link flex justify-between items-center p-2 rounded-md hover:bg-[#E6EEF3] transition-colors duration-150"
                                                   style="font-size: 0.9rem; border-radius: 0.5rem; padding: 0.5rem 0.75rem; color: var(--text-secondary);">
                                                    <span class="truncate pr-2" title="${section.title}">${section.title}</span>
                                                    <span class="text-xs font-mono px-1.5 py-0.5 bg-gray-200 text-gray-600 rounded-full flex-shrink-0">${(section.score??0).toFixed(1)}</span>
                                                </a>
                                            </li>
                                        `).join('')}
                                    </ul>
                                </nav>
                            </div>
                        </aside>
                        <!-- Main Content Area -->
                        <main class="lg:col-span-3 space-y-8 smooth-scroll-container" style="max-height: calc(100vh - 5rem); overflow-y: auto;">
                            ${(result.breakdown || []).flatMap(block => block.sections).map(section => `
                                <section id="section-${this.sanitizeId(section.title)}-${companyId}" class="detail-content-section scroll-mt-20" aria-labelledby="heading-${this.sanitizeId(section.title)}-${companyId}">
                                    <details class="section-card bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden" open>
                                        <summary class="flex justify-between items-center p-6 hover:bg-gray-50 transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-t-2xl">
                                            <h3 id="heading-${this.sanitizeId(section.title)}-${companyId}" class="font-semibold text-[#203F58]" style="font-size: 1.1rem;">${section.title}</h3>
                                            <div class="flex items-center gap-4">
                                                <span class="text-sm font-medium text-gray-600">得分 ${(section.score??0).toFixed(1)} / ${section.max_score}</span>
                                                <svg class="details-arrow w-5 h-5 text-gray-500 transition-transform duration-200" viewBox="0 0 20 20" fill="currentColor">
                                                  <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
                                                </svg>
                                            </div>
                                        </summary>
                                        <div class="px-6 pb-6 border-t border-gray-200">
                                            <!-- Progress Bar -->
                                            <div class="w-full bg-[#E9E4DF] rounded-full my-6" style="height: 6px;" role="progressbar" aria-valuenow="${section.max_score > 0 ? ((section.score ?? 0) / section.max_score * 100) : 0}" aria-valuemin="0" aria-valuemax="100" aria-labelledby="heading-${this.sanitizeId(section.title)}-${companyId}">
                                                <div class="bg-[#35637C] h-full rounded-full" style="width: ${section.max_score > 0 ? ((section.score ?? 0) / section.max_score * 100) : 0}%; transition: width 0.5s ease-in-out;"></div>
                                            </div>
                                            <!-- Criteria Details -->
                                            <div class="space-y-6">
                                                ${(section.criteria || []).map(criterion => `
                                                    <div class="border-b border-gray-100 pb-4 last:border-b-0">
                                                        <h4 class="font-semibold text-[#35637C] mb-3" style="font-weight: 600;">${criterion.title}</h4>
                                                        <div class="space-y-4">
                                                            ${(criterion.sub_criteria || []).map(sub => `
                                                                <div class="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1" style="font-size: 0.9rem;">
                                                                    <div class="col-start-1">
                                                                        <p class="text-[#1E293B] leading-relaxed"><span class="text-gray-400 mr-1">-</span>${sub.title || '項目說明遺失'}</p>
                                                                         ${sub.rationale ? `<p class="text-[#64748B] italic mt-1 pl-4 border-l-2 border-gray-200 ml-1" style="font-size: 0.85rem;">“${sub.rationale}”</p>` : ''}
                                                                    </div>
                                                                    <span class="col-start-2 font-medium text-[#1E293B] whitespace-nowrap text-right self-start pt-px">${(sub.score??0).toFixed(1)} / ${sub.max_score}</span>
                                                                </div>
                                                            `).join('')}
                                                        </div>
                                                    </div>
                                                `).join('')}
                                            </div>
                                            <!-- Section Summary Card -->
                                            ${renderSummaryCard(section)}
                                        </div>
                                    </details>
                                </section>
                            `).join('')}
                        </main>
                    </div>
                </div>
                `;
            },
            getLevelAndColor(score, forHTML = false) {
                 // ... (v6.2 logic - seems okay) ...
                 const levels = {
                     platinum: { text: '白金級', textColor: '#1f2937', bgColor: '#f3f4f6', borderColor: '#d1d5db' },
                     gold:     { text: '金級',   textColor: '#854d0e', bgColor: '#fef9c3', borderColor: '#fde68a' },
                     silver:   { text: '銀級',   textColor: '#4b5563', bgColor: '#e5e7eb', borderColor: '#d1d5db' },
                     copper:   { text: '銅級',   textColor: '#9a3412', bgColor: '#ffedd5', borderColor: '#fed7aa' },
                     default:  { text: '-',      textColor: '#4b5563', bgColor: '#e5e7eb', borderColor: '#d1d5db' }
                };
                 const cssVars = { // Kept for potential future use with CSS variables
                     platinum: { text: 'var(--platinum-text, #1f2937)', bg: 'var(--platinum-bg, #f3f4f6)' },
                     gold:     { text: 'var(--gold-text, #854d0e)',     bg: 'var(--gold-bg, #fef9c3)'     },
                     silver:   { text: 'var(--silver-text, #4b5563)',   bg: 'var(--silver-bg, #e5e7eb)'   },
                     copper:   { text: 'var(--copper-text, #9a3412)',   bg: 'var(--copper-bg, #ffedd5)'   },
                     default:  { text: 'var(--text-secondary, #4b5563)',bg: 'var(--light-bg, #e5e7eb)'    }
                 };

                let key = 'default';
                const numericScore = parseFloat(score); // Ensure score is a number
                if (isNaN(numericScore)) { key = 'default'; }
                else if (numericScore >= 90) { key = 'platinum'; }
                else if (numericScore >= 80) { key = 'gold'; }
                else if (numericScore >= 70) { key = 'silver'; }
                else if (numericScore >= 0)  { key = 'copper'; }

                const selectedLevel = levels[key];

                 // Always return direct hex values for simplicity now
                 return {
                     text: selectedLevel.text,
                     textColor: selectedLevel.textColor,
                     bgColor: selectedLevel.bgColor,
                     borderColor: selectedLevel.borderColor
                 };
            },
            sanitizeId(id) {
                 // ... (v6.2 logic - seems okay) ...
                 return (id || '')
                     .toString()
                     .toLowerCase()
                     .replace(/\s+/g, '-') // Replace spaces with hyphens
                      // Remove characters not suitable for HTML IDs/selectors
                     .replace(/[^\p{L}\p{N}_-]+/gu, ''); // Allow letters, numbers, underscore, hyphen
            }
        }
    };

    // --- PDF 生成模組 v6.2 (依據 JSON Prompt 更新) ---
    const PDFGenerator = {
        state: null,
        COLORS: {
            primary: '#203F58',     // 永續報告書用色
            media: '#F69268',       // 多元媒體用色
            accent: '#35637C',
            emphasis: '#F69268',
            background: '#F9F7F6',
            border: '#E9E4DF',
            textPrimary: '#1E293B',
            textSecondary: '#64748B',
            summaryCardBg: '#F3F4F6',
            // coverGradientStart: '#102030', // Removed
            coverGradientEnd: '#203F58',   // Use as solid background
        },

        init(appState) {
            this.state = appState;
        },

        async generate(result, buttonElement) {
            const originalButtonHTML = buttonElement.innerHTML;
            buttonElement.disabled = true;
            buttonElement.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>正在產生...`;

            try {
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });

                if (!this.state.fontData.normal) {
                    alert("錯誤：無法產生 PDF，因為必要的中文字型載入失敗。"); // Keep alert for now based on previous behavior
                    buttonElement.disabled = false;
                    return;
                }
                this.loadFontsToVFS(doc);

                // Add bookmarks before generating pages
                doc.outline.add(null, '封面', { pageNumber: 1 });
                 // Add placeholder for Summary Page bookmark
                doc.outline.add(null, '總覽', { pageNumber: 2 });
                // Add root for Detailed Scores
                 const detailRoot = doc.outline.add(null, '詳細評分'); // Don't set page number yet


                await this.createCoverPage(doc, result); // Page 1

                doc.addPage(); // Page 2 starts
                await this.createSummaryPage(doc, result);

                 // --- Start Detailed Pages ---
                 // Store the starting page number for detailRoot offset calculation
                 const detailStartPage = doc.internal.getCurrentPageInfo().pageNumber + 1;
                 this.createDetailedScorePages(doc, result, detailRoot); // Pass the root node


                // --- Finalize and Add Footers ---
                const totalPages = doc.internal.getNumberOfPages();
                for (let i = 1; i <= totalPages; i++) {
                    doc.setPage(i);
                    this._renderHeaderFooter(doc, i, totalPages, result);
                }

                 // --- Update Bookmark Page Numbers ---
                 // Bookmarks should be correct as they are added during page creation


                const companyName = Helpers.sanitizeFileName(result.company);
                doc.save(`${companyName}_AI評分報告_v6.2.pdf`); // Update version

            } catch (e) {
                console.error("PDF generation failed:", e);
                 alert('產生 PDF 時發生錯誤：' + e.message + (e.stack ? `\n${e.stack}`: '')); // Include stack trace if available
            } finally {
                buttonElement.innerHTML = originalButtonHTML;
                buttonElement.disabled = false;
            }
        },

        loadFontsToVFS(doc) {
            if (!this.state.fontData.normal) {
                 console.error("Font data is missing in state.");
                 return; // Prevent error if fontData is null
            }
            try {
                doc.addFileToVFS('NotoSansTC-Regular.ttf', this.state.fontData.normal);
                doc.addFont('NotoSansTC-Regular.ttf', 'NotoSansTC', 'normal');
                doc.addFont('NotoSansTC-Regular.ttf', 'NotoSansTC', 'bold');
                doc.setFont('NotoSansTC', 'normal'); // Set as default immediately
                console.log("Fonts loaded and set successfully.");
            } catch (e) {
                console.error("Error loading fonts to VFS:", e);
                // Maybe throw error or alert user?
                 alert("載入 PDF 字型時發生內部錯誤，匯出可能失敗。");
            }
        },

        _renderHeaderFooter(doc, pageNum, totalPages, result) {
            if (pageNum === 1) return;
            const W = doc.internal.pageSize.width;
            const H = doc.internal.pageSize.height;
            const companyName = result?.company?.replace(" (預覽)", "") || "報告"; // Fallback company name
            const currentYear = new Date().getFullYear();
            const footerText = `${companyName} ｜ ${currentYear} ｜ Page ${pageNum} of ${totalPages}`;
            doc.setFont('NotoSansTC', 'normal'); // Ensure correct font
            doc.setFontSize(9);
            Helpers.setHexColor(doc, this.COLORS.textSecondary);
            doc.text(footerText, W / 2, H - 15, { align: 'center' });
        },

        async createCoverPage(doc, result) {
            const W = doc.internal.pageSize.getWidth(); // Use getter methods
            const H = doc.internal.pageSize.getHeight();
            const M = 20; // Margin
             doc.setFont('NotoSansTC', 'normal'); // Ensure font

            // v6.2: Solid color background
            Helpers.setHexColor(doc, this.COLORS.coverGradientEnd, 'fill');
            doc.rect(0, 0, W, H, 'F');

            // --- Optional Cover Image ---
            if (Config.COVER_IMAGE_BASE64) {
                 try {
                     // Use try-catch for getImageProperties as it can fail on invalid data
                     const imgProps = doc.getImageProperties(Config.COVER_IMAGE_BASE64);
                     const imgRatio = imgProps.width / imgProps.height;
                     const pageRatio = W / H;
                     let imgWidth, imgHeight, startX, startY;

                     // Calculate image dimensions to cover/fit page
                     if (imgRatio > pageRatio) { // Image wider than page ratio
                         imgWidth = W;
                         imgHeight = W / imgRatio;
                         startX = 0;
                         startY = (H - imgHeight) / 2;
                     } else { // Image taller than page ratio
                         imgHeight = H;
                         imgWidth = H * imgRatio;
                         startY = 0;
                         startX = (W - imgWidth) / 2;
                     }
                     doc.addImage(Config.COVER_IMAGE_BASE64, 'PNG', startX, startY, imgWidth, imgHeight, undefined, 'FAST');
                 } catch (e) {
                     console.error("無法加入封面圖片:", e);
                     // Background color is already set, so just log error
                 }
            }

            // --- Title Box ---
            const titleBoxY = H / 2 - 30;
            const titleBoxHeight = 60;

            // v6.2: White box with border shadow
            doc.setFillColor(255, 255, 255); // White fill
            Helpers.setHexColor(doc, this.COLORS.border, 'draw'); // Use border color for shadow effect
            doc.setLineWidth(0.5); // Slightly thinner line width
            doc.roundedRect(M, titleBoxY, W - 2 * M, titleBoxHeight, 5, 5, 'FD'); // Fill and Draw

            // --- Titles ---
            // v6.2: Main title size 26
            doc.setFont('NotoSansTC', 'bold');
            doc.setFontSize(26);
            Helpers.setHexColor(doc, this.COLORS.textPrimary);
            const mainTitle = result.company?.replace(" (預覽)", "") || "評分報告"; // Fallback title
            doc.text(mainTitle, W / 2, titleBoxY + 20, { align: 'center', baseline: 'bottom' }); // Adjust position slightly

            // Subtitle size 18
            doc.setFont('NotoSansTC', 'normal');
            doc.setFontSize(18);
            Helpers.setHexColor(doc, this.COLORS.accent);
            doc.text("AI 智能永續報告書評分報告", W / 2, titleBoxY + 35, { align: 'center', baseline: 'top' }); // Adjust position slightly

            // --- Score Circle ---
            const score = result.totals?.final ?? 0;
            const level = UI.templates.getLevelAndColor(score, false); // Get PDF colors
            const circleY = titleBoxY + titleBoxHeight + 25;

            // Use validated colors from getLevelAndColor
            const bgColor = level.bgColor.startsWith('#') ? level.bgColor : '#e5e7eb';
            const textColor = level.textColor.startsWith('#') ? level.textColor : '#4b5563';

            doc.setFillColor(bgColor);
            doc.setDrawColor(0, 0, 0, 0); // No border
            doc.circle(W / 2, circleY, 18, 'F');

            // Score Text
            doc.setFont('NotoSansTC', 'bold');
            doc.setFontSize(24);
            Helpers.setHexColor(doc, textColor);
            doc.text(score.toFixed(1), W / 2, circleY + 1, { align: 'center', baseline: 'middle'}); // Center vertically

            // Level Text
            doc.setFont('NotoSansTC', 'normal');
            doc.setFontSize(10);
            Helpers.setHexColor(doc, textColor);
            doc.text(level.text, W / 2, circleY + 11, { align: 'center', baseline: 'middle' }); // Position below score
        },

        async _getChartImage(result) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const config = UI._getChartConfig(result);
            const numLabels = config.data.labels?.length || 0; // Handle missing labels
            canvas.width = 800; // Fixed width for high resolution
            // Ensure minimum height, adjust dynamically
            canvas.height = Math.max(300, numLabels * 35 + 60); // Increased base height and padding

             // --- PDF specific style overrides ---
             config.options.scales.y.ticks.font.size = 14;
             config.options.scales.x.ticks.font.size = 12;
             config.options.plugins.legend.display = false;
             config.options.plugins.tooltip.enabled = false;
             config.options.animation = false;
             config.options.layout.padding = { left: 15, right: 25, top: 15, bottom: 15 }; // Adjusted padding


            return new Promise((resolve, reject) => {
                 try {
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    config.options.devicePixelRatio = 2; // Render @ 2x

                    const chart = new Chart(ctx, config);

                    // Delay slightly longer to ensure full render, especially with more labels
                    setTimeout(() => {
                        try {
                            const dataUrl = canvas.toDataURL('image/png', 1.0);
                             chart.destroy();
                            resolve(dataUrl);
                        } catch (e) {
                             if(chart) chart.destroy();
                            console.error("Error converting canvas:", e);
                            reject(e);
                        }
                    }, 350); // Slightly longer delay

                 } catch (e) {
                     console.error("Error creating chart instance:", e);
                     reject(e);
                 }
            });
        },


        drawCategoryList(doc, data, x, startY, maxWidth, isDryRun = false) {
             let currentY = startY;
             const lineHeight = 1.5;
             const bullet = "•";
             const indent = 4;
             let totalHeight = 0;
             let drewItems = false; // Flag to check if any item was actually drawn

             // Check if data is fundamentally empty or invalid
             if (!data || typeof data !== 'object' || Object.keys(data).length === 0) {
                 if (!isDryRun) {
                     doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                     Helpers.setHexColor(doc, this.COLORS.textSecondary);
                     doc.text(`${bullet} 無`, x, currentY);
                 }
                 return 7; // Return minimum height
             }

            // Iterate through categories (e.g., "strengths", "improvements")
            for (const category in data) {
                const items = data[category];

                 // Check if items are valid and not just ["無具體..."]
                 const validItems = Array.isArray(items)
                     ? items.filter(item => item && typeof item === 'string' && !item.includes("無具體") && item.trim() !== "" && item.trim() !== "無")
                     : [];


                if (validItems.length > 0) {
                    drewItems = true; // Mark that we found items to draw
                    if (!isDryRun) {
                        doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                        Helpers.setHexColor(doc, this.COLORS.textPrimary);
                    }
                    for (const item of validItems) {
                        const textWidth = Math.max(1, maxWidth - indent);
                        const itemLines = doc.splitTextToSize(item, textWidth);
                        // Calculate height needed for these lines + padding
                        const itemHeight = doc.getTextDimensions(itemLines, { lineHeightFactor: lineHeight }).h + 2; // 2mm padding below item

                        if (!isDryRun) {
                            doc.text(bullet, x, currentY); // Draw bullet
                             // Draw text lines, ensuring maxWidth is respected
                            doc.text(itemLines, x + indent, currentY, { lineHeightFactor: lineHeight, maxWidth: textWidth });
                        }
                        currentY += itemHeight; // Move Y down
                        totalHeight += itemHeight; // Add to total height
                    }
                }
            } // End category loop

             // If after checking all categories, nothing was drawn, draw "無"
             if (!drewItems) {
                  if (!isDryRun) {
                      doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                      Helpers.setHexColor(doc, this.COLORS.textSecondary);
                      doc.text(`${bullet} 無`, x, startY); // Draw "無" at the original startY
                  }
                  return 7; // Return minimum height
             }


            return totalHeight; // Return the total height occupied
        },

        async createSummaryPage(doc, result) {
            let y = 30;
            const M = 20;
            const CONTENT_W = doc.internal.pageSize.getWidth() - 2 * M;
            const H = doc.internal.pageSize.getHeight();
            const B_MARGIN = 35; // Bottom margin above footer

            // --- Page Title ---
            doc.setFont('NotoSansTC', 'bold');
            doc.setFontSize(18); Helpers.setHexColor(doc, this.COLORS.textPrimary);
            doc.text("Executive Snapshot / 總覽", M, y); y += 15;

            // --- AI Summary ---
            doc.setFontSize(14); Helpers.setHexColor(doc, this.COLORS.accent);
             doc.setFont('NotoSansTC', 'bold'); // Make subtitle bold
            doc.text("AI 總評", M, y); y += 8;

            doc.setFont('NotoSansTC', 'normal'); // Reset to normal for text body
            doc.setFontSize(10); Helpers.setHexColor(doc, this.COLORS.textPrimary);
            const commentLines = doc.splitTextToSize(result.overview_comment || '無', CONTENT_W);
            doc.text(commentLines, M, y, { lineHeightFactor: 1.6 });
            y += doc.getTextDimensions(commentLines, { lineHeightFactor: 1.6 }).h + 10;

            // --- Divider ---
            Helpers.setHexColor(doc, this.COLORS.border, 'draw');
             doc.setLineWidth(0.3); // Make divider slightly thicker
            doc.line(M, y - 5, M + CONTENT_W, y - 5); y += 5;

            // --- Score Breakdown Chart ---
            doc.setFontSize(14); Helpers.setHexColor(doc, this.COLORS.accent);
            doc.setFont('NotoSansTC', 'bold'); // Make subtitle bold
            doc.text("分項得分率", M, y); y += 8;

            try {
                const chartImage = await this._getChartImage(result);
                const imgProps = doc.getImageProperties(chartImage);
                // Calculate height maintaining aspect ratio
                const chartHeight = CONTENT_W * (imgProps.height / imgProps.width);
                // Check page break *before* adding image
                if (y + chartHeight > H - B_MARGIN) {
                    doc.addPage();
                    y = 40; // Reset Y for new page with top margin
                }
                doc.addImage(chartImage, 'PNG', M, y, CONTENT_W, chartHeight);
                y += chartHeight + 10; // Space after chart
            } catch(e) {
                console.error("無法繪製 Chart 至 PDF:", e);
                // Display error message in PDF if chart fails
                 // Check page break for error message before drawing it
                 const errorTextHeight = 10; // Approximate height for error text
                 if (y + errorTextHeight > H - B_MARGIN) { doc.addPage(); y = 40; }
                 doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                 Helpers.setHexColor(doc, '#DC2626'); // Red color for error
                doc.text("無法載入評分圖表。", M, y); y += errorTextHeight; // Use estimated height
            }

            // --- Strengths & Improvements ---
            const colWidth = CONTENT_W / 2 - 8; // Width per column (-16 for gap)
            const gap = 16; // Gap between columns

            // Use "總覽" key for summary page
            const strengthsSummary = result.strengths?.總覽 ? { "strengths": result.strengths.總覽 } : null;
            const improvementsSummary = result.improvements?.總覽 ? { "improvements": result.improvements.總覽 } : null;


            const sHeight = this.drawCategoryList(doc, strengthsSummary, 0, 0, colWidth, true);
            const iHeight = this.drawCategoryList(doc, improvementsSummary, 0, 0, colWidth, true);
            // Estimate needed height: Title + Margin + Max List Height
            const neededListHeight = 12 + Math.max(sHeight, iHeight);

            // Check page break before drawing divider and lists
             if (y + neededListHeight > H - B_MARGIN) {
                 doc.addPage(); y = 40; // Reset Y with top margin
             }


            // --- Divider ---
            Helpers.setHexColor(doc, this.COLORS.border, 'draw');
            doc.setLineWidth(0.3); // Consistent divider thickness
            doc.line(M, y - 5, M + CONTENT_W, y - 5); y += 5; // Space below divider

            // --- Draw Titles and Lists ---
            const listStartY = y + 8; // Y where list items start (below title)
            doc.setFontSize(12); doc.setFont('NotoSansTC', 'bold');

            // Strengths Title
            Helpers.setHexColor(doc, '#15803d'); // Green
            doc.text("主要優勢", M, y);
            // Strengths List
            this.drawCategoryList(doc, strengthsSummary, M, listStartY, colWidth);

            // Improvements Title
             const improvementsX = M + colWidth + gap;
            Helpers.setHexColor(doc, '#be123c'); // Red
            doc.text("改善建議", improvementsX, y);
            // Improvements List
            this.drawCategoryList(doc, improvementsSummary, improvementsX, listStartY, colWidth);

             // Update y to the bottom of the taller list + margin (no longer needed as this is the end of the page)
             // y = listStartY + Math.max(sHeight, iHeight) + 10; // Removed

        },

        // =========================================================================
        // =                      【v6.2 改良版】createDetailedScorePages                   =
        // =========================================================================
        createDetailedScorePages(doc, result, parentBookmark) {
            const W = doc.internal.pageSize.getWidth();
            const H = doc.internal.pageSize.getHeight();
            const M = 20; // Margin
            let y = 30; // Current Y position on the page

             // Define minimum space needed before footer
             const FOOTER_SPACE = 25; // page number area + breathing room

            // Helper to check and potentially add a new page
            const checkPageBreak = (needed = 10) => {
                if (y + needed > H - FOOTER_SPACE) {
                    doc.addPage();
                    y = 30; // Reset Y for the new page
                    return true;
                }
                return false;
            };

            const breakdown = result.breakdown || [];
            breakdown.forEach((module) => {
                const isReportModule = module.id === 'report';
                const moduleTitle = isReportModule ? '永續報告書 (60%)' : '多元媒體應用與內容品質 (40%)';
                const sections = module.sections || [];
                const moduleTotalSections = sections.length;
                let moduleSectionCounter = 0;

                // Add module bookmark pointing to its title page (which will be the *next* page)
                 const moduleNode = doc.outline.add(parentBookmark, moduleTitle, { pageNumber: doc.internal.getCurrentPageInfo().pageNumber + 1 });

                // --- Module Title Page ---
                doc.addPage();
                y = 50; // Y position for title page content
                doc.setFont('NotoSansTC', 'bold');
                doc.setFontSize(20);
                const moduleColor = isReportModule ? this.COLORS.primary : this.COLORS.media;
                Helpers.setHexColor(doc, moduleColor);
                doc.text(moduleTitle, W / 2, y, { align: 'center' });
                y += 12;
                // Divider line below title
                Helpers.setHexColor(doc, this.COLORS.border, 'draw');
                doc.setLineWidth(0.5);
                doc.line(M, y, W - M, y);
                // y is not advanced further; next section starts on a new page


                // --- Iterate through sections ---
                sections.forEach((section) => {
                    moduleSectionCounter++;

                    // Start each section on a fresh page
                    doc.addPage();
                    y = 30; // Reset Y for the new section page

                    // Add section bookmark for the current page
                    doc.outline.add(moduleNode, section.title, { pageNumber: doc.internal.getCurrentPageInfo().pageNumber });


                    // --- Section Header ---
                    const headerColor = isReportModule ? this.COLORS.primary : this.COLORS.media;

                    // v6.2: Top guide color strip
                    Helpers.setHexColor(doc, headerColor, 'fill');
                    doc.rect(M, y, W - 2 * M, 1.5, 'F'); // Thin strip at the top
                    y += 3.5; // Space below strip

                    // Header background block
                    Helpers.setHexColor(doc, headerColor, 'fill');
                    doc.roundedRect(M, y, W - 2 * M, 10, 2, 2, 'F');
                    doc.setFont('NotoSansTC', 'bold');
                    doc.setFontSize(13);
                    Helpers.setHexColor(doc, '#FFFFFF'); // White text
                    doc.text(section.title || '未命名章節', M + 6, y + 7); // Section title with fallback
                    // Score text
                    const scoreText = `${(section.score ?? 0).toFixed(1)} / ${section.max_score ?? 'N/A'}`;
                    doc.text(scoreText, W - M - 6, y + 7, { align: 'right' });
                    y += 14; // Space below header block

                    // --- Section Summary ---
                    doc.setFont('NotoSansTC', 'normal');
                    doc.setFontSize(10);
                    Helpers.setHexColor(doc, this.COLORS.textPrimary);
                    const summaryText = `此章節評估企業於「${section.title || '?'}」面向的表現與揭露狀況。以下為細項評分與分析結果。`;
                    const lines = doc.splitTextToSize(summaryText, W - 2 * M);
                     if (checkPageBreak(doc.getTextDimensions(lines, { lineHeightFactor: 1.5 }).h + 5)) { /* y reset */ } // Check break before drawing
                    doc.text(lines, M, y, { lineHeightFactor: 1.5 });
                    y += doc.getTextDimensions(lines, { lineHeightFactor: 1.5 }).h + 5;

                    // --- Progress Bar ---
                    const scoreValue = section.score ?? 0;
                    const maxValue = section.max_score ?? 0;
                    const pct = maxValue > 0 ? (scoreValue / maxValue) : 0;
                     if (checkPageBreak(8)) { /* y reset */ } // Check break before drawing
                    Helpers.setHexColor(doc, this.COLORS.border, 'fill'); // Background bar
                    doc.roundedRect(M, y, W - 2 * M, 2, 1, 1, 'F');
                    Helpers.setHexColor(doc, headerColor, 'fill'); // Progress color
                    doc.roundedRect(M, y, (W - 2 * M) * pct, 2, 1, 1, 'F');
                    y += 8; // Space below progress bar

                    // --- Criteria Groups ---
                    (section.criteria || []).forEach((criterion) => {
                         // Dry run to get height first
                        const neededHeight = this.drawCriterionGroup(doc, criterion, M, y, W - 2 * M, checkPageBreak, true);
                         // Check if the *entire group* plus spacing fits
                         if (checkPageBreak(neededHeight + 8)) {
                             // If it broke, y is already reset to 30.
                             // Optionally redraw section header/title here if needed for context on new page
                         }
                        // Now draw the group
                        const actualHeight = this.drawCriterionGroup(doc, criterion, M, y, W - 2 * M, checkPageBreak, false);
                        y += actualHeight + 8; // Space after the group
                    });

                    // --- Divider before Summary Card (only if content exists) ---
                    if (y > 40) { // Add divider only if there's significant content
                        if (checkPageBreak(5 + 4)) { /* y reset */ } // Check space for divider and margin
                        Helpers.setHexColor(doc, this.COLORS.border, 'draw');
                        doc.setLineWidth(0.2);
                        doc.line(M, y - 4, W - M, y - 4); // Line above margin
                        y += 4; // Margin below divider
                    } else {
                         y += 4; // Ensure some space if no divider
                     }


                    // --- Summary Card ---
                    const strengthsData = result.strengths && result.strengths[section.title];
                    const improvementsData = result.improvements && result.improvements[section.title];
                    const estHeight = this.drawSummaryCard(doc, M, y, W - 2 * M, strengthsData, improvementsData, true);
                    const breathingSpaceHeight = 6;
                    const totalFooterHeight = estHeight + 2 + breathingSpaceHeight + 10; // Card + space + block + bottom margin

                    if (checkPageBreak(totalFooterHeight)) {
                         // y is reset if page break occurs
                    }

                    const cardHeight = this.drawSummaryCard(doc, M, y, W - 2 * M, strengthsData, improvementsData, false);
                    y += cardHeight + 2; // Space below card

                    // v6.2: Footer breathing space block
                    Helpers.setHexColor(doc, this.COLORS.background, 'fill');
                    doc.rect(M, y, W - 2 * M, breathingSpaceHeight, 'F');

                    // v6.2: Section index text
                    doc.setFont('NotoSansTC', 'normal');
                    doc.setFontSize(8);
                    Helpers.setHexColor(doc, this.COLORS.textSecondary);
                    const sectionIndexText = `${isReportModule ? '永續報告書' : '多元媒體'} ${moduleSectionCounter.toString().padStart(2, '0')} / ${moduleTotalSections.toString().padStart(2, '0')}`;
                    doc.text(sectionIndexText, W - M - 4, y + breathingSpaceHeight - 2, { align: 'right', baseline: 'bottom' });

                    // y is implicitly ready for the next section's page break or the end
                    // No need to add extra y += here as the loop continues or finishes

                }); // End sections.forEach
            }); // End breakdown.forEach (modules)
        },


        // =========================================================================
        // =                      【v6.2 改良版】drawCriterionGroup                     =
        // =========================================================================
        drawCriterionGroup(doc, criterion, x, y, width, checkPageBreak, isDryRun = false) {
            let startY = y;
            const indent = 4;
            const lineSpacing = 1.5;
            const quoteLineSpacing = 1.6;

            // --- Calculate total height for dry run ---
             let calculatedHeight = 0;
             calculatedHeight += 6; // Space below title

             doc.setFont('NotoSansTC', 'bold'); doc.setFontSize(11); // Title font
             // Assuming title itself won't wrap significantly or cause breaks

             (criterion.sub_criteria || []).forEach((sub, idx) => {
                 const quoteText = sub.rationale ? `“${sub.rationale}”` : '';
                 const textWidth = width - indent - 25;

                 doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10); // Sub-title font
                 const titleLines = doc.splitTextToSize(sub.title || '', textWidth);
                 const titleHeight = doc.getTextDimensions(titleLines, { lineHeightFactor: lineSpacing }).h;
                 calculatedHeight += titleHeight + 2.5; // v6.2 space after title

                 if (quoteText) {
                     doc.setFontSize(9); // Quote font
                     const quoteLines = doc.splitTextToSize(quoteText, width - indent - 20);
                     const quoteHeight = doc.getTextDimensions(quoteLines, { lineHeightFactor: quoteLineSpacing }).h;
                     calculatedHeight += quoteHeight + 5; // v6.2 space after quote
                 } else {
                     calculatedHeight += 2.5; // Space even if no quote
                 }

                 if (idx < (criterion.sub_criteria.length - 1)) {
                     calculatedHeight += 6; // v6.2 space after separator
                 } else {
                     calculatedHeight += 3; // Space after last item
                 }
             });

            if (isDryRun) {
                return calculatedHeight; // Return calculated height for page break check
            }


            // --- Actual Drawing ---
            doc.setFont('NotoSansTC', 'bold');
            doc.setFontSize(11);
            Helpers.setHexColor(doc, this.COLORS.primary);
            doc.text(criterion.title || '評分標準', x, y); // Title with fallback
            y += 6; // Space below title

            (criterion.sub_criteria || []).forEach((sub, idx) => {
                 // Check if *this specific sub-criterion* needs a page break before drawing
                 let subNeededHeight = 0;
                 const quoteText = sub.rationale ? `“${sub.rationale}”` : '';
                 const textWidth = width - indent - 25;
                 doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                 const titleLines = doc.splitTextToSize(sub.title || '', textWidth);
                 const titleHeight = doc.getTextDimensions(titleLines, { lineHeightFactor: lineSpacing }).h;
                 subNeededHeight += titleHeight + 2.5; // Space after title
                 if (quoteText) {
                    doc.setFontSize(9);
                    const quoteLines = doc.splitTextToSize(quoteText, width - indent - 20);
                    subNeededHeight += doc.getTextDimensions(quoteLines, { lineHeightFactor: quoteLineSpacing }).h + 5; // Space after quote
                 } else {
                    subNeededHeight += 2.5;
                 }
                 if (idx < (criterion.sub_criteria.length - 1)) subNeededHeight += 6; else subNeededHeight += 3; // Space for separator/end

                 if (checkPageBreak(subNeededHeight)) {
                      // y is reset by checkPageBreak
                      // Redraw the main criterion title as context on the new page
                      doc.setFont('NotoSansTC', 'bold'); doc.setFontSize(11);
                      Helpers.setHexColor(doc, this.COLORS.primary);
                      doc.text(`${criterion.title || '評分標準'} (續)`, x, y);
                      y += 6;
                      // Don't reset startY here, as height calculation is per-group draw call
                 }


                const subScore = `${(sub.score ?? 0).toFixed(1)} / ${sub.max_score}`;
                // Draw Title and Score
                doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(10);
                Helpers.setHexColor(doc, this.COLORS.textPrimary);
                doc.text("-", x, y);
                doc.text(titleLines, x + indent, y, { lineHeightFactor: lineSpacing, maxWidth: textWidth });
                doc.text(subScore, x + width, y, { align: 'right' });
                y += titleHeight + 2.5; // v6.2 space

                // Draw Rationale (Quote)
                if (quoteText) {
                    doc.setFont('NotoSansTC', 'normal'); doc.setFontSize(9);
                    Helpers.setHexColor(doc, this.COLORS.textSecondary);
                    const quoteLines = doc.splitTextToSize(quoteText, width - indent - 20);
                     const quoteHeight = doc.getTextDimensions(quoteLines, { lineHeightFactor: quoteLineSpacing }).h;
                    doc.text(quoteLines, x + indent + 2, y, { lineHeightFactor: quoteLineSpacing, maxWidth: width - indent - 20 });
                    y += quoteHeight + 5; // v6.2 space
                } else {
                     y += 2.5; // Keep consistent spacing
                }

                // Draw Separator Line
                if (idx < (criterion.sub_criteria.length - 1)) {
                    Helpers.setHexColor(doc, this.COLORS.border, 'draw');
                    doc.setLineWidth(0.1);
                    doc.line(x, y, x + width, y);
                    y += 6; // v6.2 space
                } else {
                    y += 3; // Smaller space after the last item
                }
            });

            return y - startY; // Return the actual height used by this group on the current page
        },


        // =========================================================================
        // =                      【v6.2 改良版】drawSummaryCard                   =
        // =========================================================================
        drawSummaryCard(doc, x, y, width, strengths, improvements, isDryRun = false) {
            const P = 8; // Padding inside card
            // v6.2: Adjusted column widths and positions
            const gap = 8; // Gap between columns, ensure it's considered
            const availableWidth = width - 2 * P - gap; // Total width available for text content
            const leftColW = availableWidth * 0.52;
            const rightColW = availableWidth * 0.48;
            const separatorX = x + P + leftColW + gap / 2; // Position of the vertical separator line
            const rightColTextX = separatorX + gap / 2; // Starting X for right column text


            const TITLE_H = 6; // Estimated height for the title line itself (adjust if needed)
            const TITLE_MARGIN_BOTTOM = 6; // Space below title line

            // --- Calculate Height ---
            const sHeight = this.drawCategoryList(doc, { "strengths": strengths }, 0, 0, leftColW, true);
            const iHeight = this.drawCategoryList(doc, { "improvements": improvements }, 0, 0, rightColW, true);
            const listHeight = Math.max(sHeight, iHeight, 5); // Ensure a minimum list height (e.g., for "無")
            // v6.2: Increase bottom margin calculation: P(top)+Title+Margin+ListHeight+P(bottom)+Extra(10)
            const cardHeight = P + TITLE_H + TITLE_MARGIN_BOTTOM + listHeight + P + 10;

            if (isDryRun) return cardHeight;

            // --- Actual Drawing ---
            // Background and Border
            Helpers.setHexColor(doc, this.COLORS.summaryCardBg, 'fill');
            Helpers.setHexColor(doc, this.COLORS.border, 'draw');
            doc.setLineWidth(0.2); // Use consistent line width
            doc.roundedRect(x, y, width, cardHeight, 4, 4, 'FD'); // Fill and Draw border

            // Separator Line (v6.2 position)
            // Helpers.setHexColor(doc, this.COLORS.border, 'draw'); // Already set
            doc.line(separatorX, y + P * 0.5, separatorX, y + cardHeight - P * 0.5); // Extend line slightly

            const titleY = y + P + TITLE_H * 0.7; // Y position for titles (adjust baseline)

            // --- Left Column: Strengths ---
            doc.setFont('NotoSansTC', 'bold');
            doc.setFontSize(11);
            Helpers.setHexColor(doc, this.COLORS.accent); // Blue for strengths title
            doc.text("✅ 主要優勢", x + P, titleY);
            // Draw list using calculated width and position
            this.drawCategoryList(doc, { "strengths": strengths }, x + P, titleY + TITLE_MARGIN_BOTTOM, leftColW);

            // --- Right Column: Improvements ---
            doc.setFont('NotoSansTC', 'bold'); // Ensure font is reset if drawCategoryList changes it
            doc.setFontSize(11);
            Helpers.setHexColor(doc, this.COLORS.emphasis); // Orange for improvements title
            // Draw title using calculated position
            doc.text("⚠️ 改善建議", rightColTextX, titleY); // Start text at rightColTextX
            // Draw list using calculated width and position
            this.drawCategoryList(doc, { "improvements": improvements }, rightColTextX, titleY + TITLE_MARGIN_BOTTOM, rightColW);


            return cardHeight;
        }
    };

    // --- 應用程式邏輯模組 ---
    const Logic = {
        state: null,
        init(appState) { this.state = appState; },
        handleFiles(files) {
            // ... (v6.2 logic with coupled file/website removal) ...
            if (this.state.isMockMode && this.state.currentStep !== 1) {
                // If in mock mode and adding files after step 1, reset first
                 this.resetApp();
                 this.state.isMockMode = true; // Re-enter mock mode state if needed
                 document.body.classList.add('preview-mode');
                 if (UI.elements.previewBanner) UI.elements.previewBanner.style.display = 'block';
                 // We are now at step 1, let the normal flow handle the file
            }

            let fileAdded = false;
            for (const file of files) {
                // Basic validation
                 if (!file.type.startsWith('application/pdf')) {
                     console.warn(`Skipped non-PDF file: ${file.name}`);
                     continue; // Skip non-PDF files
                 }
                 if (file.size > 50 * 1024 * 1024) { // Example: 50MB limit
                     alert(`檔案 "${file.name}" 過大 (超過 50MB)，已略過。`);
                     continue;
                 }


                const fileObject = { file: file, id: 'file-' + Date.now() + Math.random(), name: file.name, size: file.size };
                 // Prevent duplicate file names
                 if (!this.state.files.some(f => f.name === fileObject.name)) {
                    this.state.files.push(fileObject);
                    const companyName = file.name.replace(/\.pdf$/i, '').trim();
                    // Add website entry only if company name doesn't exist
                    if (!this.state.websites.some(w => w.company === companyName)) {
                        this.addWebsite(companyName, ''); // Add with empty URL
                    }
                    fileAdded = true;
                 } else {
                     console.warn(`File "${fileObject.name}" already exists, skipped.`);
                     // Optionally notify the user about duplicates
                 }
            }
             if (fileAdded) {
                UI.renderFileList();
                UI.renderWebsiteList();
             } else if (files.length > 0) {
                 // If files were provided but none were added (e.g., all duplicates or wrong type)
                 alert("未加入任何新檔案。檔案可能已存在、非 PDF 格式或過大。");
             }
        },
        addWebsite(companyName = '', url = '') {
            // Prevent adding empty company name entries implicitly? Or allow and validate later?
             // Let's allow for now, validation happens in startProcessing
            const websiteObject = { id: 'site-' + Date.now() + Math.random(), company: companyName, url: url };
            this.state.websites.push(websiteObject);
            UI.renderWebsiteList();
        },
        async startProcessing() {
            // ... (v6.2 logic - includes FormData metadata) ...
            if (this.state.isProcessing) return;

            if (this.state.isMockMode) {
                UI.changeStep(3);
                const progressText = document.getElementById('progress-text');
                if(progressText) progressText.textContent = '正在載入預覽資料...';
                setTimeout(() => {
                    this.state.results = MOCK_RESULTS;
                    UI.renderResults();
                    UI.changeStep(4);
                }, 1000); // Shorter delay for mock
                return;
            }

            const matchedData = [];
            let hasEmptyCompanyName = false;
            let filesWithoutMatch = [...this.state.files]; // Track files that don't get matched

             this.state.websites.forEach(site => {
                 const trimmedCompany = site.company.trim();
                 if (!trimmedCompany) {
                     hasEmptyCompanyName = true;
                     return; // Skip this entry, error handled below
                 }
                 // Find matching file (case-insensitive and trim spaces)
                 const fileIndex = filesWithoutMatch.findIndex(f =>
                    f.name.replace(/\.pdf$/i, '').trim().toLowerCase() === trimmedCompany.toLowerCase()
                 );

                 if (fileIndex > -1) {
                     const matchedFile = filesWithoutMatch.splice(fileIndex, 1)[0]; // Remove from unmatched list
                     matchedData.push({
                         file: matchedFile.file,
                         company: trimmedCompany,
                         url: site.url?.trim() || 'N/A'
                     });
                 } else {
                      console.warn(`Website entry "${trimmedCompany}" has no matching PDF file.`);
                      // Optionally, decide if this should be an error or just a warning
                 }
             });


            if (hasEmptyCompanyName) {
                 alert('錯誤：企業名稱欄位不能是空的。請填寫所有企業名稱。');
                 return;
             }
             if (matchedData.length === 0) {
                 alert('錯誤：找不到任何 PDF 檔案與輸入的企業名稱相符。請確保 PDF 檔名（不含 .pdf）與企業名稱欄位完全一致。');
                 return;
             }
             // Optional: Warn about files that were uploaded but had no matching website entry
             if (filesWithoutMatch.length > 0) {
                 console.warn(`以下 PDF 檔案沒有對應的企業名稱/網站項目，將不會處理：`, filesWithoutMatch.map(f => f.name));
                 // alert(`警告：部分上傳的 PDF 檔案 (${filesWithoutMatch.map(f => f.name).join(', ')}) 沒有對應的企業名稱項目，將不會處理。`);
             }


            this.state.isProcessing = true;
            UI.changeStep(3);
            const progressText = document.getElementById('progress-text');

            try {
                const formData = new FormData();
                // Append files under 'files' key
                matchedData.forEach(item => {
                    formData.append('files', item.file, item.file.name);
                });
                // Append metadata as a single JSON string under 'metadata' key
                 const metadata = matchedData.map(item => ({ company: item.company, website_url: item.url }));
                 formData.append('metadata', JSON.stringify(metadata));

                if(progressText) progressText.textContent = `正在準備上傳 ${matchedData.length} 個檔案...`;

                const response = await fetch(`${Config.API_BASE_URL}/score_reports`, { // Ensure endpoint matches backend
                    method: 'POST',
                    body: formData
                });

                if(progressText) progressText.textContent = '後端處理中，請稍候... (這可能需要數分鐘)';

                if (!response.ok) {
                    let errorData = { detail: `伺服器錯誤: ${response.status} ${response.statusText}` };
                    try {
                        const errJson = await response.json();
                         // More specific error structure checks
                         if (errJson && errJson.detail) {
                             errorData.detail = Array.isArray(errJson.detail)
                                 ? errJson.detail.map(d => d.msg || JSON.stringify(d)).join('; ') // Handle Pydantic validation errors
                                 : errJson.detail;
                         } else if (errJson && errJson.message) {
                            errorData.detail = errJson.message;
                         }
                    } catch (e) {
                         // Response was not JSON, maybe plain text or HTML error page
                         const textResponse = await response.text();
                         console.error("Non-JSON error response:", textResponse);
                         // Keep the generic status text error
                     }
                    throw new Error(errorData.detail);
                }

                let data;
                try { data = await response.json(); }
                catch(e) {
                    console.error("無法解析伺服器回應 JSON:", e);
                     const textResponse = await response.text().catch(() => "無法讀取回應內容");
                     console.error("伺服器回應 (Text):", textResponse);
                    throw new Error("伺服器回應格式錯誤。");
                }

                this.state.results = Array.isArray(data) ? data : [data]; // Ensure results is always an array
                UI.renderResults();
                UI.changeStep(4);

            } catch (error) {
                console.error('處理失敗:', error);
                 // Display a user-friendly message
                 alert(`處理過程中發生錯誤：\n${error.message}\n請檢查您的檔案、網路連線，或稍後再試。`);
                UI.changeStep(2); // Go back to step 2
            } finally {
                this.state.isProcessing = false;
                if(progressText) progressText.textContent = ''; // Clear progress text
            }
        },
        resetApp() {
            // ... (v6.2 logic - seems okay) ...
            const wasInMockMode = this.state.isMockMode;
            Object.assign(this.state, {
                currentStep: 1,
                files: [],
                websites: [],
                results: [],
                isProcessing: false,
                isMockMode: false // Explicitly reset mock mode on general reset
            });
             // Reset UI elements
            UI.renderFileList();
            UI.renderWebsiteList();
             if (UI.elements.resultsContainer) UI.elements.resultsContainer.innerHTML = '';
            const fileInput = document.getElementById('file-input');
            if(fileInput) fileInput.value = ''; // Clear file input selection

             // Reset connection status message
            const statusDiv = document.getElementById('connection-status');
            if (statusDiv) {
                statusDiv.classList.add('hidden');
                statusDiv.textContent = '';
                statusDiv.classList.remove('text-green-600', 'text-red-600');
            }
             // Ensure preview mode banner is hidden if exiting
             if (UI.elements.previewBanner) UI.elements.previewBanner.style.display = 'none';
             document.body.classList.remove('preview-mode');


            UI.changeStep(1); // Go back to step 1
        },
        startPreviewMode() {
             // Reset state before entering preview mode to ensure clean slate
             this.resetApp();
            this.state.isMockMode = true;
            document.body.classList.add('preview-mode');
            if (UI.elements.previewBanner) UI.elements.previewBanner.style.display = 'block';
            // Pre-populate mock file *after* resetting
            const mockFile = new File(["mock content"], "永續模範企業 (預覽).pdf", {type: "application/pdf"});
            Logic.handleFiles([mockFile]); // This adds file and website entry
            UI.changeStep(2); // Move to step 2 after setup
        },
        exitPreviewMode() {
            this.state.isMockMode = false; // Set flag first
            document.body.classList.remove('preview-mode');
            if (UI.elements.previewBanner) UI.elements.previewBanner.style.display = 'none';
            this.resetApp(); // Reset everything and go to step 1
        },
        async testBackendConnection() {
            // ... (v6.2 logic - seems okay) ...
             const btn = document.getElementById('test-connection-btn');
            const statusDiv = document.getElementById('connection-status');
            if (!btn || !statusDiv) return;

            const textSpan = btn.querySelector('#test-text');
            const originalText = textSpan ? textSpan.textContent : '測試後端連接';

            btn.disabled = true;
            if(textSpan) textSpan.textContent = "測試中...";
            statusDiv.classList.remove('hidden', 'text-green-600', 'text-red-600');
            statusDiv.textContent = '正在連接...';

            try {
                // Add timeout to fetch request
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

                const response = await fetch(`${Config.API_BASE_URL}/health`, { signal: controller.signal });
                 clearTimeout(timeoutId); // Clear timeout if fetch completes

                if (response.ok) {
                    const data = await response.json();
                    statusDiv.textContent = `✅ 連接成功: ${data.message || '服務正常'}`;
                    statusDiv.classList.add('text-green-600');
                } else {
                    let errorMsg = `伺服器回應錯誤: ${response.status} ${response.statusText}`;
                    try { const errData = await response.json(); errorMsg = `伺服器錯誤: ${errData.detail || response.statusText}`; } catch (e) { /* ignore */ }
                    throw new Error(errorMsg);
                }
            } catch (error) {
                 console.error("Connection test failed:", error);
                 let displayError = error.message;
                 if (error.name === 'AbortError') {
                     displayError = "連接超時。";
                 } else if (error instanceof TypeError) {
                    displayError = "網路錯誤或 CORS 設定問題。";
                 }
                statusDiv.textContent = `❌ 連接失敗: ${displayError} 請確認後端 (${Config.API_BASE_URL}) 已啟動且可訪問。`;
                statusDiv.classList.add('text-red-600');
            } finally {
                btn.disabled = false;
                 if(textSpan) textSpan.textContent = originalText;
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
            fontData: { normal: null },
            isProcessing: false,
            isMockMode: false,
        },
        async init() {
            UI.init(this.state);
            PDFGenerator.init(this.state);
            Logic.init(this.state);
            this.bindEvents();
            UI.updateStepUI();
            await this.loadInitialFonts();
        },
        async loadInitialFonts() {
            const fontStatusElement = UI.elements.fontStatusContainer;
            if (fontStatusElement) {
                // Using a simpler spinner or just text
                fontStatusElement.innerHTML = `<span class="spinner mr-2"></span>正在載入 PDF 所需字型...`;
                 fontStatusElement.className = 'text-sm text-gray-500 flex items-center gap-2';
            }

            const fontBase64 = await Helpers.loadFontAsBase64(Config.FONT_PATHS.normal);

            if (fontBase64 && !fontBase64.error) {
                this.state.fontData.normal = fontBase64;
                if (fontStatusElement) {
                    fontStatusElement.innerHTML = `✅ PDF 所需字型已成功載入。`;
                    fontStatusElement.className = 'text-sm text-green-600 flex items-center gap-2';
                }
            } else {
                 this.state.fontData.normal = null;
                if (fontStatusElement) {
                    const errorMsg = (fontBase64 && fontBase64.error) || "未知錯誤";
                    fontStatusElement.innerHTML = `<span class="error-icon mr-1.5">⚠️</span><span>警告：PDF 字型載入失敗...<br><small>${errorMsg}</small></span>`;
                    fontStatusElement.className = 'text-sm text-red-600 flex items-start gap-2';
                }
            }
        },
        bindEvents() {
            // Use optional chaining for safety
            document.getElementById('next-step1')?.addEventListener('click', () => UI.changeStep(2));
            document.getElementById('prev-step2')?.addEventListener('click', () => UI.changeStep(1));
            document.getElementById('next-step2')?.addEventListener('click', () => Logic.startProcessing());
            document.getElementById('back-to-start')?.addEventListener('click', () => Logic.resetApp());

            const dropArea = document.getElementById('drop-area');
            const fileInput = document.getElementById('file-input');
            if (dropArea && fileInput) {
                dropArea.addEventListener('click', () => fileInput.click());
                fileInput.addEventListener('change', (e) => Logic.handleFiles(e.target.files));

                const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    dropArea.addEventListener(eventName, preventDefaults, false);
                    dropArea.addEventListener(eventName, (e) => {
                        dropArea.classList.toggle('dragover', ['dragenter', 'dragover'].includes(e.type));
                    }, false);
                });
                dropArea.addEventListener('drop', (e) => {
                    dropArea.classList.remove('dragover'); // Ensure dragover is removed on drop
                    Logic.handleFiles(e.dataTransfer.files);
                }, false);
            }

            document.getElementById('add-website-btn')?.addEventListener('click', () => Logic.addWebsite('', ''));
            document.getElementById('start-preview-btn')?.addEventListener('click', () => Logic.startPreviewMode());
            document.getElementById('exit-preview-btn')?.addEventListener('click', () => Logic.exitPreviewMode());
            document.getElementById('test-connection-btn')?.addEventListener('click', () => Logic.testBackendConnection());
        }
    };

    // --- 完整的範例資料 (MOCK_RESULTS v6.2 - Restored sub_criteria) ---
    const MOCK_RESULTS = [
        {
            "company": "永續模範企業 (預覽)",
            "overview_comment": "永續模範企業在各個面向都展現了卓越的領導力。報告書結構清晰、數據透明，並有效利用多元媒體進行溝通，是業界的標竿。(此為預覽模式 v6.2)",
            "strengths": {
                "總覽": ["重大性議題矩陣圖呈現清晰", "策略面完整揭露", "合理等級確信報告", "治理結構清晰", "永續專區內容豐富", "高品質永續影片"],
                "完整性": ["重大性議題矩陣圖呈現清晰，且各章節均能有效連結回重大議題。", "策略面完整揭露了短、中、長期的永續發展藍圖與目標。"],
                "可信度": ["附有會計師出具的合理等級確信報告，大幅提升資訊可信度。", "治理結構清晰，已揭露高階薪酬與 ESG 績效的連結性。"],
                "溝通性": ["報告書圖文並茂，排版優良。", "提供完整的英文版報告書。"],
                "組織永續專區": ["官方網站設有內容豐富的永續專區，且更新頻繁、資訊即時。"], // Moved from media section for clarity if needed
                "多元媒體展現": ["網站中包含多部高品質的永續專題影片，溝通方式多元。"] // Moved from media section
            },
            "improvements": {
                "總覽": ["增加互動式導覽功能", "強化利害關係人回應客製化", "增加負面訊息說明"],
                "溝通性": ["報告書架構雖清晰，但可考慮增加互動式導覽功能，提升讀者查閱體驗。", "部分利害關係人議合的回應較為制式，可再強化客製化與深度。"],
                "可信度": ["雖已揭露多數績效，但可考慮增加過去負面訊息的說明與改善措施，展現更高的透明度。"]
            },
            "breakdown": [
              {
                "id": "report",
                "sections": [
                  {
                    "title": "完整性", "score": 38.0, "max_score": 40.0,
                    "criteria": [
                       {"title": "重大性議題", "score": 8.0, "max_score": 8.0, "sub_criteria": [
                           {"title": "是否清楚列出或呈現重大性議題分析之矩陣圖或其他圖表，且清楚標明各項議題的種類", "max_score": 2.0, "score": 2.0, "rationale": "報告書第XX頁呈現了清晰的重大性議題矩陣圖。"},
                           {"title": "是否清楚說明組織重大性議題分析之過程與方法", "max_score": 2.0, "score": 2.0, "rationale": "有詳細說明分析過程。"},
                           {"title": "是否有呈現出重大性議題在報告書中的連結性", "max_score": 2.0, "score": 2.0, "rationale": "各章節均能連結回重大議題。"},
                           {"title": "是否清楚說明重大性議題對於組織的意義", "max_score": 2.0, "score": 2.0, "rationale": "說明了議題對營運的影響。"}
                       ]},
                       {"title": "利害關係人共融", "score": 6.0, "max_score": 6.0, "sub_criteria": [
                           {"title": "是否清楚列出組織的利害關係人之種類與意義", "max_score": 1.0, "score": 1.0, "rationale": "已鑑別主要利害關係人。"},
                           {"title": "是否清楚說明各種利害關係人議合之方法", "max_score": 2.0, "score": 2.0, "rationale": "提供了多元的溝通管道列表。"},
                           {"title": "是否清楚說明各種利害關係人關注之議題", "max_score": 1.0, "score": 1.0, "rationale": "已歸納各方關注焦點。"},
                           {"title": "是否清楚說明組織針對各項議題的因應之道", "max_score": 2.0, "score": 2.0, "rationale": "針對關注議題提出具體回應。"}
                       ]},
                       {"title": "策略", "score": 12.0, "max_score": 12.0, "sub_criteria": [
                           {"title": "報告書中是否有說明永續對組織的重要性與意義(價值鏈的呈現)", "max_score": 2.0, "score": 2.0, "rationale": "開篇即闡述永續價值主張。"},
                           {"title": "報告書中是否有揭露組織營運相關之內外部環境分析", "max_score": 3.0, "score": 3.0, "rationale": "包含 PESTEL 分析。"},
                           {"title": "報告書中是否有揭露組織對於環境、社會、治理等面向的發展原則與管理機制(長期策略)", "max_score": 3.0, "score": 3.0, "rationale": "揭露了2030永續藍圖。"},
                           {"title": "是否有在各個面向或是各類重大性議題說明組織未來改善目標(中期策略)", "max_score": 2.0, "score": 2.0, "rationale": "設定了具體的3年改善目標。"},
                           {"title": "針對各項重大性議題是否有設定隔年度之量化或是質化目標(短期策略)", "max_score": 2.0, "score": 2.0, "rationale": "各章節末皆有次年度KPI。"}
                       ]},
                       {"title": "組織介紹", "score": 2.0, "max_score": 2.0, "sub_criteria": [
                           {"title": "揭露資訊：主要產品與服務、財務績效、地理分布、員工資訊、整體環境與組織營運之關聯性等", "max_score": 2.0, "score": 2.0, "rationale": "公司概況章節完整揭露。"}
                       ]},
                       {"title": "重大永續規範執行及資訊揭露", "score": 10.0, "max_score": 12.0, "sub_criteria": [
                           {"title": "氣候相關財務揭露(TCFD)", "max_score": 3.0, "score": 3.0, "rationale": "設有獨立 TCFD 報告書章節。"},
                           {"title": "永續會計準則委員會準則(SASB)", "max_score": 3.0, "score": 3.0, "rationale": "提供 SASB 指標索引。"},
                           {"title": "自然相關財務揭露(TNFD)", "max_score": 3.0, "score": 2.0, "rationale": "已初步導入 TNFD 框架進行評估。"},
                           {"title": "國際財務報導準則(IFRS) S1,S2揭露", "max_score": 3.0, "score": 2.0, "rationale": "已開始進行IFRS S1/S2的鑑別與準備。"}
                       ]}
                    ]
                  },
                  {
                    "title": "可信度", "score": 32.5, "max_score": 35.0,
                    "criteria": [
                       {"title": "管理流程", "score": 10.0, "max_score": 10.0, "sub_criteria": [
                            {"title": "報告揭露採用之指引與準則", "max_score": 1.0, "score": 1.0, "rationale": "明確遵循 GRI Standards。"},
                            {"title": "是否揭露報告書主要負責單位", "max_score": 1.0, "score": 1.0, "rationale": "載明永續發展委員會為負責單位。"},
                            {"title": "報告書的管理方式", "max_score": 4.0, "score": 4.0, "rationale": "說明了資料收集與審核流程。"},
                            {"title": "針對各項重大性議題皆說明管理方針", "max_score": 4.0, "score": 4.0, "rationale": "各議題均有對應的管理方針。"}
                       ]},
                       {"title": "利害關係人回應", "score": 4.5, "max_score": 5.0, "sub_criteria": [
                            {"title": "針對利害關係人關注之議題，組織是否實際回應議題，並提出相對應之作為、策略與規劃等政策", "max_score": 2.0, "score": 2.0, "rationale": "均有對應的回應與作為。"},
                            {"title": "組織是否有針對組織鑑別出之實質性議題進行回應，並提出相對應之策略與作為", "max_score": 3.0, "score": 2.5, "rationale": "大部分議題有回應，少部分較為籠統。"}
                       ]},
                       {"title": "治理", "score": 4.0, "max_score": 5.0, "sub_criteria": [
                            {"title": "是否有說明組織組織針對永續報告的責任單位", "max_score": 1.0, "score": 1.0, "rationale": "已說明責任單位。"},
                            {"title": "報告書是否有說明董事會的薪酬與永續績效的連結性", "max_score": 2.0, "score": 2.0, "rationale": "已揭露高階薪酬與ESG 績效連結。"},
                            {"title": "報告書是否有揭露組織組織的風險與可能之機會(因應之道)", "max_score": 1.0, "score": 1.0, "rationale": "風險管理章節有說明。"},
                            {"title": "組織績效指標管理方針是否與組織永續原則一致", "max_score": 1.0, "score": 0.0, "rationale": "未明確說明其一致性。"}
                       ]},
                       {"title": "績效", "score": 4.0, "max_score": 5.0, "sub_criteria": [
                            {"title": "績效之揭露是否完整(重大性議題涵蓋經濟、環境與社會，是否有質化的說明與數據)", "max_score": 2.0, "score": 2.0, "rationale": "績效數據揭露完整。"},
                            {"title": "重大性議題是否有量化的圖表說明", "max_score": 1.0, "score": 1.0, "rationale": "多數採用量化圖表。"},
                            {"title": "是否有揭露過去負面訊息", "max_score": 1.0, "score": 0.0, "rationale": "未見揭露負面訊息。"},
                            {"title": "績效的呈現是否易懂", "max_score": 1.0, "score": 1.0, "rationale": "圖表清晰易懂。"}
                       ]},
                       {"title": "保證/確信", "score": 10.0, "max_score": 10.0, "sub_criteria": [
                            {"title": "是否已建立永續資訊編制內部控制制度及相關流程", "max_score": 2.0, "score": 2.0, "rationale": "已建立相關內控制度。"},
                            {"title": "永續資訊編制內部控制制度及其內部稽核執行情形說明", "max_score": 3.0, "score": 3.0, "rationale": "說明了內稽的執行狀況。"},
                            {"title": "是否有外部第三方獨立保證/確信之佐證資料", "max_score": 2.0, "score": 2.0, "rationale": "附有會計師出具之確信報告。"},
                            {"title": "外部保證是否有說明保證等級、範疇與方法(中度/有限等級者最多得2分，高度/合理等級者最多可得3分)", "max_score": 3.0, "score": 3.0, "rationale": "提供了合理等級的確信。"}
                       ]}
                    ]
                  },
                  {
                    "title": "溝通性", "score": 21.5, "max_score": 25.0,
                    "criteria": [
                       {"title": "展現", "score": 9.0, "max_score": 10.0, "sub_criteria": [
                            {"title": "版面是否圖表與文字說明比例恰當，內容清晰且易於閱讀", "max_score": 3.0, "score": 3.0, "rationale": "圖文並茂，排版優良。"},
                            {"title": "具有英文版報告書", "max_score": 3.0, "score": 3.0, "rationale": "提供完整的英文版報告書。"},
                            {"title": "展現創新的資訊呈現方式", "max_score": 2.0, "score": 1.0, "rationale": "有使用資訊圖表，但較少創新互動設計。"},
                            {"title": "報告書之份量是否適當(頁數120-150頁為參考範圍)", "max_score": 2.0, "score": 2.0, "rationale": "頁數適中(130頁)。"}
                       ]},
                       {"title": "利害關係人共融", "score": 4.5, "max_score": 5.0, "sub_criteria": [
                            {"title": "組織永續報告書是否公開下載", "max_score": 1.0, "score": 1.0, "rationale": "官網提供公開下載。"},
                            {"title": "是否有說明利害關係人議合(溝通資訊)的方法", "max_score": 2.0, "score": 2.0, "rationale": "已說明溝通方法。"},
                            {"title": "利害關係人議合的結果，組織是否公開揭露其相對應的回應與作為", "max_score": 2.0, "score": 1.5, "rationale": "有揭露，但部分回應較為制式。"}
                       ]},
                       {"title": "架構", "score": 8.0, "max_score": 10.0, "sub_criteria": [
                            {"title": "是否清楚整理並呈現本年度的亮點作為報告書的總結", "max_score": 3.0, "score": 3.0, "rationale": "報告書前段有 Highlight 整理。"},
                            {"title": "完整的索引設計(包括GRI, SASB及其他重要規範等)", "max_score": 3.0, "score": 3.0, "rationale": "附錄提供完整 GRI/SASB 索引。"},
                            {"title": "報告書附有清楚的連結，使讀者可透過網頁的說明獲得更細節的資訊", "max_score": 2.0, "score": 1.0, "rationale": "部分連結可點擊，但並非全部。"},
                            {"title": "架構呈現完整易于查閱", "max_score": 2.0, "score": 1.0, "rationale": "目錄清晰，但缺乏互動式導覽。"}
                       ]}
                    ]
                  }
                ]
              },
              {
                "id": "media",
                "sections": [
                  {
                    "title": "組織永續專區", "score": 3.0, "max_score": 3.0,
                    "criteria": [
                       {"title": "組織永續專區", "score": 3.0, "max_score": 3.0, "sub_criteria": [ // Renamed criterion title to match section
                            {"title": "設置組織永續專區", "max_score": 0.5, "score": 0.5, "rationale": "官網設有永續專區。"},
                            {"title": "是否將組織永續專區連結設於首頁", "max_score": 0.5, "score": 0.5, "rationale": "首頁有明顯連結。"},
                            {"title": "是否提供報告書下載", "max_score": 0.5, "score": 0.5, "rationale": "提供歷年報告書下載。"},
                            {"title": "是否有網站地圖", "max_score": 0.5, "score": 0.5, "rationale": "網站頁尾提供網站地圖。"},
                            {"title": "站內搜尋引擎", "max_score": 0.5, "score": 0.5, "rationale": "具備站內搜尋功能。"},
                            {"title": "是否將組織永續專區分類且內容充實", "max_score": 0.5, "score": 0.5, "rationale": "內容分類清晰，資訊豐富。"}
                       ]}
                    ]
                  },
                  {
                    "title": "網頁管理與即時更新", "score": 4.0, "max_score": 4.0,
                    "criteria": [
                       {"title": "網頁管理與即時更新", "score": 4.0, "max_score": 4.0, "sub_criteria": [
                            {"title": "判斷依據：由最新消息觀察網頁是否為最新訊息、是否即時更新", "max_score": 4.0, "score": 4.0, "rationale": "最新消息更新頻繁，資訊即時。"}
                       ]}
                    ]
                  },
                  {
                    "title": "電子版報告書與關鍵資訊連結", "score": 3.0, "max_score": 4.0,
                    "criteria": [
                       {"title": "電子版報告書與關鍵資訊連結", "score": 3.0, "max_score": 4.0, "sub_criteria": [
                            {"title": "按照永續報告定義，須符合環境、社會與治理(ESG)以及供應鏈管理等四項議題之揭露", "max_score": 4.0, "score": 3.0, "rationale": "網站內容涵蓋ESG各面向。"}
                       ]}
                    ]
                  },
                  {
                    "title": "多元媒體展現", "score": 4.0, "max_score": 4.0,
                    "criteria": [
                       {"title": "多元媒體展現", "score": 4.0, "max_score": 4.0, "sub_criteria": [
                            {"title": "文字說明", "max_score": 1.0, "score": 1.0, "rationale": "文字清晰易懂。"},
                            {"title": "圖表說明", "max_score": 1.0, "score": 1.0, "rationale": "使用多樣化的互動圖表。"},
                            {"title": "使用影片", "max_score": 1.0, "score": 1.0, "rationale": "網站中包含多部高品質的永續專題影片。"},
                            {"title": "互動式網頁", "max_score": 1.0, "score": 1.0, "rationale": "提供了互動式的數據查詢頁面。"}
                       ]}
                    ]
                  },
                  {
                    "title": "溝通回饋管道與社群網絡互動", "score": 4.0, "max_score": 4.0,
                    "criteria": [
                       {"title": "溝通回饋管道與社群網絡互動", "score": 4.0, "max_score": 4.0, "sub_criteria": [
                            {"title": "線上回饋機制之應用(網路填寫或連結至電子信箱)", "max_score": 1.0, "score": 1.0, "rationale": "提供線上聯絡表單。"},
                            {"title": "線上互動式機制之應用", "max_score": 1.0, "score": 1.0, "rationale": "設有利害關係人專區。"},
                            {"title": "社交網站之應用", "max_score": 1.0, "score": 1.0, "rationale": "活躍於 LinkedIn, Facebook 等社群平台。"},
                            {"title": "提供訂閱電子報", "max_score": 1.0, "score": 1.0, "rationale": "提供永續電子報訂閱服務。"}
                       ]}
                    ]
                  }
                ]
              }
            ],
           "totals": { "report": 55.2, "media": 36.1, "final": 91.3 } // Example final scores, ensure consistency
        }
    ];


    // --- 應用程式啟動 ---
    App.init();
});

