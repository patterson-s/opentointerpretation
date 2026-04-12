/* ── Section router ─────────────────────────────────────────────────────────
 * Hash-based navigation: #companies | #models | #countries
 * Adding a new section requires:
 *   1. A <section id="section-<name>"> in index.html
 *   2. A <a href="#<name>"> nav link in index.html
 *   No JS changes needed for basic "coming soon" sections.
 */

const SECTIONS = ['companies', 'models', 'countries'];
const DEFAULT_SECTION = 'companies';

function activateSection(name) {
  if (!SECTIONS.includes(name)) name = DEFAULT_SECTION;

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(a => {
    const target = a.getAttribute('href').slice(1); // strip '#'
    a.classList.toggle('active', target === name);
  });

  // Show/hide sections
  SECTIONS.forEach(s => {
    const el = document.getElementById(`section-${s}`);
    if (el) el.classList.toggle('active', s === name);
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


/* ── Init ───────────────────────────────────────────────────────────────────*/
loadCompanyList().catch(err => {
  console.error('Failed to load companies:', err);
  const ul = document.getElementById('company-list');
  if (ul) ul.innerHTML = '<li style="padding:.75rem 1rem;color:#ef4444">Failed to load</li>';
});
