/* ── Section router ─────────────────────────────────────────────────────────
 * Hash-based navigation: #companies | #models | #countries | #analysis
 * Adding a new section requires:
 *   1. A <section id="section-<name>"> in index.html
 *   2. A <a href="#<name>"> nav link in index.html
 *   No JS changes needed for basic "coming soon" sections.
 */

const SECTIONS = ['companies', 'models', 'countries', 'analysis', 'historical'];
const DEFAULT_SECTION = 'companies';

function activateSection(name) {
  console.log(`Activating section: ${name}`); // Debug
  
  if (!SECTIONS.includes(name)) {
    console.log(`Section ${name} not found, defaulting to ${DEFAULT_SECTION}`);
    name = DEFAULT_SECTION;
  }

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(a => {
    const target = a.getAttribute('href').slice(1); // strip '#'
    a.classList.toggle('active', target === name);
  });

  // Show/hide sections
  SECTIONS.forEach(s => {
    const el = document.getElementById(`section-${s}`);
    if (el) {
      el.classList.toggle('active', s === name);
      console.log(`Section ${s}: ${el.classList.contains('active') ? 'ACTIVE' : 'inactive'}`); // Debug
    } else {
      console.log(`Section element for ${s} not found!`); // Debug
    }
  });
}

function onHashChange() {
  const hash = location.hash.slice(1) || DEFAULT_SECTION;
  activateSection(hash);
}

window.addEventListener('hashchange', onHashChange);
onHashChange(); // run on load


/* ── Company list ───────────────────────────────────────────────────────────*/

let allCompanies = [];
let activeChart = null;

async function loadCompanyList() {
  const res = await fetch('/api/companies');
  if (!res.ok) throw new Error(`Failed to fetch companies: ${res.status}`);
  allCompanies = await res.json();
  renderCompanyList(allCompanies);
}

function renderCompanyList(companies) {
  const ul = document.getElementById('company-list');
  ul.innerHTML = '';
  companies.forEach(c => {
    const li = document.createElement('li');
    li.textContent = c.display_name;
    li.dataset.id = c.id;
    li.addEventListener('click', () => selectCompany(c.id, li));
    ul.appendChild(li);
  });
}

document.getElementById('company-search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  const filtered = allCompanies.filter(c => c.display_name.toLowerCase().includes(q));
  renderCompanyList(filtered);
});

function selectCompany(id, liEl) {
  // Highlight selection
  document.querySelectorAll('#company-list li').forEach(el => el.classList.remove('selected'));
  if (liEl) liEl.classList.add('selected');
  loadCompanyDetail(id);
}


/* ── Company detail ─────────────────────────────────────────────────────────*/

async function loadCompanyDetail(id) {
  const panel = document.getElementById('company-detail');
  panel.innerHTML = '<div class="detail-empty">Loading…</div>';

  const res = await fetch(`/api/companies/${id}`);
  if (!res.ok) {
    panel.innerHTML = `<div class="detail-empty">Error loading company (${res.status})</div>`;
    return;
  }
  const data = await res.json();
  renderCompanyDetail(panel, data);
}

function renderCompanyDetail(panel, d) {
  // Destroy existing chart if any
  if (activeChart) {
    activeChart.destroy();
    activeChart = null;
  }

  const handleHtml = d.hf_handle
    ? `<span class="company-handle">huggingface.co/${d.hf_handle}</span>`
    : '';

  panel.innerHTML = `
    <div class="company-header">
      <div class="company-name">${escHtml(d.display_name)}</div>
      ${handleHtml}
    </div>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Country HQ</div>
        <div class="stat-value small">${escHtml(d.country_hq || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Models</div>
        <div class="stat-value">${d.model_count}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">HuggingFace</div>
        <div class="stat-value">${d.hf_count}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Closed-source</div>
        <div class="stat-value">${d.closed_count}</div>
      </div>
    </div>

    <div class="section-title">License Distribution</div>
    <div class="license-chart-wrap">
      <canvas id="license-chart"></canvas>
    </div>
  `;

  renderLicenseChart(d.license_distribution);
}

function renderLicenseChart(dist) {
  if (!dist || dist.length === 0) return;

  const labels = dist.map(r => r.slug);
  const counts = dist.map(r => r.count);

  // Color mapping: known open slugs get accent blue; unknown/bespoke get gray
  const OPEN_SLUGS = new Set(['apache-2.0', 'mit', 'cc-by-4.0', 'cc-by-nc-4.0', 'gpl-3.0', 'lgpl-3.0', 'bsd-3-clause', 'bsd-2-clause']);
  const colors = labels.map(slug => {
    if (slug === 'unknown') return '#d1d5db';
    if (OPEN_SLUGS.has(slug)) return '#2563eb';
    return '#60a5fa';
  });

  const ctx = document.getElementById('license-chart').getContext('2d');
  activeChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: colors,
        borderRadius: 3,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.x} models`
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { precision: 0, font: { family: 'monospace', size: 11 } },
          grid: { color: '#f3f4f6' }
        },
        y: {
          ticks: { font: { family: 'monospace', size: 11 } },
          grid: { display: false }
        }
      }
    }
  });
}


/* ── Utilities ──────────────────────────────────────────────────────────────*/

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


/* ── Analysis section ────────────────────────────────────────────────────────*/

let activeAnalysisChart = null;

async function loadAnalysisMetric(metric) {
  const contentEl = document.getElementById('analysis-content');
  contentEl.innerHTML = '<div class="analysis-loading">Loading analysis data...</div>';

  try {
    let data, renderFn;
    
    switch (metric) {
      case 'model-releases-by-country':
        data = await fetch('/api/analysis/model-releases-by-country').then(r => r.json());
        renderFn = renderModelReleasesByCountry;
        break;
      case 'model-releases-by-company':
        data = await fetch('/api/analysis/model-releases-by-company').then(r => r.json());
        renderFn = renderModelReleasesByCompany;
        break;
      default:
        throw new Error('Unknown metric');
    }

    renderFn(contentEl, data);
    loadNotesForMetric(metric); // Load notes for this metric
  } catch (err) {
    console.error('Failed to load analysis:', err);
    contentEl.innerHTML = '<div class="analysis-error">Failed to load analysis data</div>';
  }
}

function renderModelReleasesByCountry(container, data) {
  renderBarChart(container, data, 'country', 'Model Releases by Country');
}

function renderModelReleasesByCompany(container, data) {
  renderBarChart(container, data, 'company_name', 'Model Releases by Company');
}

function renderBarChart(container, data, labelField, title) {
  // Destroy existing chart if any
  if (activeAnalysisChart) {
    activeAnalysisChart.destroy();
    activeAnalysisChart = null;
  }

  // Check if data is empty
  if (!data || data.length === 0) {
    container.innerHTML = '<div class="analysis-error">No data available for this metric</div>';
    return;
  }

  const labels = data.map(item => item[labelField]);
  const counts = data.map(item => item.model_count);

  // Create canvas for chart
  container.innerHTML = `<div class="analysis-chart-wrap"><h3>${title}</h3><canvas id="analysis-chart"></canvas></div>`;
  const ctx = document.getElementById('analysis-chart').getContext('2d');

  activeAnalysisChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Number of Models',
        data: counts,
        backgroundColor: '#2563eb',
        borderRadius: 3,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y} models`
          }
        }
      },
      scales: {
        x: {
          ticks: { font: { family: 'monospace', size: 11 } },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0, font: { family: 'monospace', size: 11 } },
          grid: { color: '#f3f4f6' }
        }
      }
    }
  });
}

/* ── Notes system ──────────────────────────────────────────────────────────*/

function loadNotesForMetric(metric) {
  const notes = localStorage.getItem(`analysis-notes-${metric}`) || '';
  const notesEl = document.getElementById('analysis-notes');
  if (notesEl) {
    notesEl.value = notes;
    
    // Setup auto-save
    notesEl.oninput = () => {
      localStorage.setItem(`analysis-notes-${metric}`, notesEl.value);
    };
  }
}

// Setup analysis navigation
function setupAnalysisNavigation() {
  const metricsList = document.querySelector('.analysis-metrics-list');
  if (!metricsList) return;

  metricsList.addEventListener('click', (e) => {
    const item = e.target.closest('.analysis-metric-item');
    if (!item) return;

    // Update active state
    metricsList.querySelectorAll('.analysis-metric-item').forEach(el => {
      el.classList.remove('active');
    });
    item.classList.add('active');

    const metric = item.dataset.metric;
    loadAnalysisMetric(metric);
  });

  // Load default metric
  const defaultMetric = metricsList.querySelector('.analysis-metric-item.active')?.dataset.metric || 'model-releases-by-country';
  if (defaultMetric) {
    loadAnalysisMetric(defaultMetric);
  }
}

/* ── Init ───────────────────────────────────────────────────────────────────*/
loadCompanyList().catch(err => {
  console.error('Failed to load companies:', err);
  const ul = document.getElementById('company-list');
  if (ul) ul.innerHTML = '<li style="padding:.75rem 1rem;color:#ef4444">Failed to load</li>';
});

// Setup analysis section when it becomes active
const setupAnalysis = () => {
  console.log('setupAnalysis called'); // Debug
  const section = document.getElementById('section-analysis');
  console.log('Analysis section element:', section); // Debug
  if (section) {
    console.log('Analysis section classes:', section.classList); // Debug
    console.log('Is active:', section.classList.contains('active')); // Debug
  }
  if (section && section.classList.contains('active')) {
    console.log('Setting up analysis navigation...'); // Debug
    setupAnalysisNavigation();
  }
};

// Check on initial load and hash changes
setupAnalysis();
window.addEventListener('hashchange', () => {
  console.log('Hash changed, setting up analysis...'); // Debug
  setTimeout(setupAnalysis, 50); // Small delay to allow section to become active
});

// Debug: log section activation
console.log('Analysis section setup complete');


/* ── Historical section ──────────────────────────────────────────────────────*/

const HIST_PALETTE = [
  '#2563eb', '#16a34a', '#dc2626', '#d97706', '#7c3aed',
  '#db2777', '#0891b2', '#65a30d', '#9333ea', '#ea580c',
];

let activeHistoricalChart = null;

// Pivot flat [{month, <keyField>, model_count}] rows into
// { months: [...], series: [{label, data: [...]}] }
// Optionally limit to the top N series by total count.
function pivotHistoricalData(rows, keyField, topN) {
  const monthSet = new Set();
  const totals = {};
  rows.forEach(r => {
    monthSet.add(r.month);
    totals[r[keyField]] = (totals[r[keyField]] || 0) + Number(r.model_count);
  });

  let keys = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
  if (topN) keys = keys.slice(0, topN);
  const keySet = new Set(keys);

  const months = [...monthSet].sort();

  // Build lookup: month+key → count
  const lookup = {};
  rows.forEach(r => {
    if (keySet.has(r[keyField])) {
      lookup[`${r.month}|${r[keyField]}`] = Number(r.model_count);
    }
  });

  const series = keys.map(k => ({
    label: k,
    data: months.map(m => lookup[`${m}|${k}`] || 0),
  }));

  return { months, series };
}

async function loadHistoricalMetric(metric) {
  const contentEl = document.getElementById('historical-content');
  contentEl.innerHTML = '<div class="analysis-loading">Loading historical data...</div>';

  try {
    let data;
    switch (metric) {
      case 'historical-total':
        data = await fetch('/api/analysis/historical-total').then(r => r.json());
        renderHistoricalLineChart(contentEl, data);
        break;
      case 'historical-by-company':
        data = await fetch('/api/analysis/historical-by-company').then(r => r.json());
        renderHistoricalStackedChart(contentEl, data, 'company_name', 'Model Releases by Company', 10);
        break;
      case 'historical-by-country':
        data = await fetch('/api/analysis/historical-by-country').then(r => r.json());
        renderHistoricalStackedChart(contentEl, data, 'country', 'Model Releases by Country', null);
        break;
      default:
        throw new Error('Unknown historical metric');
    }
    loadHistoricalNotes(metric);
  } catch (err) {
    console.error('Failed to load historical data:', err);
    contentEl.innerHTML = '<div class="analysis-error">Failed to load historical data</div>';
  }
}

function renderHistoricalLineChart(container, data) {
  if (activeHistoricalChart) { activeHistoricalChart.destroy(); activeHistoricalChart = null; }
  if (!data || data.length === 0) {
    container.innerHTML = '<div class="analysis-error">No data available</div>';
    return;
  }

  const labels = data.map(r => r.month);
  const counts = data.map(r => Number(r.model_count));

  container.innerHTML = `
    <div class="analysis-chart-wrap">
      <h3>Historical Model Releases (Total)</h3>
      <canvas id="historical-chart"></canvas>
    </div>`;

  const ctx = document.getElementById('historical-chart').getContext('2d');
  activeHistoricalChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Models Released',
        data: counts,
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37,99,235,0.10)',
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} models` } },
      },
      scales: {
        x: {
          ticks: { font: { family: 'monospace', size: 10 }, maxRotation: 45 },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0, font: { family: 'monospace', size: 11 } },
          grid: { color: '#f3f4f6' },
        },
      },
    },
  });
}

function renderHistoricalStackedChart(container, data, keyField, title, topN) {
  if (activeHistoricalChart) { activeHistoricalChart.destroy(); activeHistoricalChart = null; }
  if (!data || data.length === 0) {
    container.innerHTML = '<div class="analysis-error">No data available</div>';
    return;
  }

  const { months, series } = pivotHistoricalData(data, keyField, topN);

  container.innerHTML = `
    <div class="analysis-chart-wrap">
      <h3>${title}</h3>
      <canvas id="historical-chart"></canvas>
    </div>`;

  const ctx = document.getElementById('historical-chart').getContext('2d');
  activeHistoricalChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: months,
      datasets: series.map((s, i) => ({
        label: s.label,
        data: s.data,
        backgroundColor: HIST_PALETTE[i % HIST_PALETTE.length],
        borderRadius: 2,
        borderSkipped: false,
      })),
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { font: { family: 'monospace', size: 11 }, boxWidth: 12 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}` } },
      },
      scales: {
        x: {
          stacked: true,
          ticks: { font: { family: 'monospace', size: 10 }, maxRotation: 45 },
          grid: { display: false },
        },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { precision: 0, font: { family: 'monospace', size: 11 } },
          grid: { color: '#f3f4f6' },
        },
      },
    },
  });
}

function loadHistoricalNotes(metric) {
  const notes = localStorage.getItem(`historical-notes-${metric}`) || '';
  const notesEl = document.getElementById('historical-notes');
  if (notesEl) {
    notesEl.value = notes;
    notesEl.oninput = () => {
      localStorage.setItem(`historical-notes-${metric}`, notesEl.value);
    };
  }
}

let historicalNavReady = false;

function setupHistoricalNavigation() {
  if (historicalNavReady) return;
  const metricsList = document.getElementById('historical-metrics-list');
  if (!metricsList) return;

  metricsList.addEventListener('click', e => {
    const item = e.target.closest('.analysis-metric-item');
    if (!item) return;
    metricsList.querySelectorAll('.analysis-metric-item').forEach(el => el.classList.remove('active'));
    item.classList.add('active');
    loadHistoricalMetric(item.dataset.metric);
  });

  historicalNavReady = true;
  const defaultItem = metricsList.querySelector('.analysis-metric-item.active');
  if (defaultItem) loadHistoricalMetric(defaultItem.dataset.metric);
}

const setupHistorical = () => {
  const section = document.getElementById('section-historical');
  if (section && section.classList.contains('active')) {
    setupHistoricalNavigation();
  }
};

setupHistorical();
window.addEventListener('hashchange', () => setTimeout(setupHistorical, 50));
