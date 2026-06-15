/* ── Section router ─────────────────────────────────────────────────────────
 * Hash-based navigation: #companies | #models | #countries | #analysis
 * Adding a new section requires:
 *   1. A <section id="section-<name>"> in index.html
 *   2. A <a href="#<name>"> nav link in index.html
 *   No JS changes needed for basic "coming soon" sections.
 */

const SECTIONS = ['home', 'companies', 'models', 'countries', 'licenses', 'analysis', 'map'];
const DEFAULT_SECTION = 'home';

function activateSection(name) {
  if (!SECTIONS.includes(name)) name = DEFAULT_SECTION;

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(a => {
    const target = a.getAttribute('href').slice(1);
    a.classList.toggle('active', target === name);
  });

  // Show/hide sections
  SECTIONS.forEach(s => {
    const el = document.getElementById(`section-${s}`);
    if (el) el.classList.toggle('active', s === name);
  });
}

function onHashChange() {
  let hash = location.hash.slice(1) || DEFAULT_SECTION;
  // Redirect legacy #historical to #analysis + switch sub-tab
  if (hash === 'historical') {
    location.replace('#analysis');
    activateSection('analysis');
    switchAnalysisSubtab('historical');
    return;
  }
  activateSection(hash);
}

window.addEventListener('hashchange', onHashChange);
onHashChange(); // run on load


/* ── Company list ───────────────────────────────────────────────────────────*/

let allCompanies = [];

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
  const handleHtml = d.hf_handle
    ? `<span class="company-handle">huggingface.co/${d.hf_handle}</span>`
    : '';

  const licDistRows = (d.license_distribution || []).map(r => {
    const pct = d.model_count > 0 ? ((r.count / d.model_count) * 100).toFixed(1) : '0.0';
    return `<tr>
      <td><a href="#licenses" class="license-link" data-slug="${escHtml(r.slug)}">${escHtml(r.slug)}</a></td>
      <td style="text-align:right;font-family:var(--font-mono);color:var(--gray-600)">${r.count}</td>
      <td style="text-align:right;font-family:var(--font-mono);color:var(--gray-400)">${pct}%</td>
    </tr>`;
  }).join('');

  const typoDistRows = (d.typology_distribution || []).map(r => {
    const pct = d.model_count > 0 ? ((r.count / d.model_count) * 100).toFixed(1) : '0.0';
    return `<tr>
      <td><span class="typology-type-badge typology-${escHtml(r.typology_type)}">${escHtml(r.typology_type)}</span></td>
      <td style="text-align:right;font-family:var(--font-mono);color:var(--gray-600)">${r.count}</td>
      <td style="text-align:right;font-family:var(--font-mono);color:var(--gray-400)">${pct}%</td>
    </tr>`;
  }).join('');

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

    <div class="section-title">Typology Breakdown</div>
    <table class="license-dist-table">
      <thead>
        <tr>
          <th style="text-align:left">Type</th>
          <th style="text-align:right">Count</th>
          <th style="text-align:right">%</th>
        </tr>
      </thead>
      <tbody>${typoDistRows}</tbody>
    </table>

    <div class="section-title" style="margin-top:1.5rem">License Distribution</div>
    <table class="license-dist-table">
      <thead>
        <tr>
          <th style="text-align:left">License</th>
          <th style="text-align:right">Count</th>
          <th style="text-align:right">%</th>
        </tr>
      </thead>
      <tbody>${licDistRows}</tbody>
    </table>

    <div class="section-title" style="margin-top:1.5rem">Models</div>
    <div id="company-models-panel">
      <div class="detail-empty" style="height:auto;padding:1rem 0">Loading models…</div>
    </div>
  `;

  // Wire up license slug click-throughs
  panel.querySelectorAll('.license-link').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const slug = a.dataset.slug;
      location.hash = 'licenses';
      setTimeout(() => {
        const li = document.querySelector(`#license-list li[data-slug="${CSS.escape(slug)}"]`);
        if (li) {
          selectLicense(slug, li);
          li.scrollIntoView({ block: 'nearest' });
        } else {
          loadLicenseDetail(slug);
        }
      }, 80);
    });
  });

  loadCompanyModels(d.id);
}

async function loadCompanyModels(companyId) {
  const panel = document.getElementById('company-models-panel');
  if (!panel) return;
  try {
    const data = await fetch(`/api/models?company_id=${companyId}&limit=50&offset=0`).then(r => r.json());
    renderCompanyModelsTable(panel, data.models, data.total);
  } catch (err) {
    panel.innerHTML = '<div class="detail-empty" style="height:auto">Failed to load models</div>';
  }
}

function renderCompanyModelsTable(panel, models, total) {
  if (!models || models.length === 0) {
    panel.innerHTML = '<div class="detail-empty" style="height:auto;padding:.5rem 0">No models recorded.</div>';
    return;
  }

  function modelRow(m, variantLabel, extraClass = '') {
    const date   = m.release_date ? String(m.release_date).slice(0, 10) : '—';
    const params = m.num_parameters != null ? `${Number(m.num_parameters).toFixed(1)}B` : '—';
    return `<tr data-model-id="${escHtml(m.model_id)}" class="clickable-row model-variant-row ${extraClass}">
      <td class="model-id-cell" title="${escHtml(m.model_id)}">${escHtml(variantLabel)}</td>
      <td><span class="models-license-tag">${escHtml(m.license_slug || '—')}</span></td>
      <td>${escHtml(m.modality || '—')}</td>
      <td>${m.typology_type ? `<span class="typology-type-badge typology-${escHtml(m.typology_type)}">${escHtml(m.typology_type)}</span>` : '—'}</td>
      <td class="num-cell">${params}</td>
      <td class="num-cell">${date}</td>
    </tr>`;
  }

  function renderUnit(unit) {
    if (unit.type === 'row') {
      return modelRow(unit.model, unit.variantLabel);
    }
    if (unit.type === 'subgroup') {
      const subRows = unit.children.map(c => modelRow(c.model, c.variantLabel, 'model-subgroup-child')).join('');
      return `<tr class="model-subgroup-header"><td colspan="6">${escHtml(unit.label)}</td></tr>${subRows}`;
    }
    if (unit.type === 'group') {
      const inner = unit.children.map(renderUnit).join('');
      return `<tr class="model-group-header"><td colspan="6">${escHtml(unit.label)}</td></tr>${inner}`;
    }
    return '';
  }

  const grouped = groupModelsByPrefix(models);
  const rows = grouped.map(renderUnit).join('');

  const note = total > 50
    ? `<div class="company-models-note">Showing 50 of ${total.toLocaleString()} models</div>`
    : '';
  panel.innerHTML = `${note}
    <div class="company-models-wrap">
      <table class="models-table company-models-table">
        <thead><tr>
          <th>Model</th><th>License</th><th>Modality</th><th>Type</th><th>Params</th><th>Released</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  // Wire row clicks to model detail
  panel.querySelectorAll('tr[data-model-id]').forEach(tr => {
    tr.addEventListener('click', () => {
      const companyDetailPanel = document.getElementById('company-detail');
      loadCompanyModelDetail(tr.dataset.modelId, companyDetailPanel);
    });
  });
}

async function loadCompanyModelDetail(modelId, returnPanel) {
  const savedContent = returnPanel.innerHTML;
  returnPanel.innerHTML = '<div class="detail-empty">Loading…</div>';

  try {
    const res = await fetch(`/api/models/${modelId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();

    returnPanel.innerHTML = '';

    const backBar = document.createElement('div');
    backBar.className = 'company-model-detail-back';
    backBar.innerHTML = `<button class="models-back-btn">← Back</button>`;
    backBar.querySelector('button').addEventListener('click', () => {
      returnPanel.innerHTML = savedContent;
      // Re-wire the rows after restoring
      returnPanel.querySelectorAll('tr[data-model-id]').forEach(tr => {
        tr.addEventListener('click', () => loadCompanyModelDetail(tr.dataset.modelId, returnPanel));
      });
    });

    const detailContent = document.createElement('div');
    detailContent.className = 'company-model-detail-content';
    renderModelDetail(detailContent, d);

    returnPanel.appendChild(backBar);
    returnPanel.appendChild(detailContent);
  } catch (err) {
    console.error('Failed to load model detail:', err);
    returnPanel.innerHTML = savedContent;
  }
}


/* ── Utilities ──────────────────────────────────────────────────────────────*/

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function stripOrgPrefix(modelId) {
  const slash = modelId.indexOf('/');
  return slash >= 0 ? modelId.slice(slash + 1) : modelId;
}

// Tokenize a model name for prefix comparison.
// Splits on - and _ but preserves the original string for display.
function tokenizeName(name) {
  return name.split(/[-_]/);
}

// Group an array of model objects by longest common token prefix.
// Returns an array of render-units:
//   { type: 'group',    label, children: [ row | subgroup ] }
//   { type: 'subgroup', label, children: [ row ] }
//   { type: 'row',      model, variantLabel }
function groupModelsByPrefix(models) {
  // Build a map: bare name → model
  const entries = models.map(m => ({
    model: m,
    bare: stripOrgPrefix(m.model_id),
    tokens: tokenizeName(stripOrgPrefix(m.model_id)),
  }));

  // Find the longest shared prefix (by token count) among a set of entries.
  function longestSharedPrefix(items) {
    if (items.length < 2) return 0;
    const first = items[0].tokens;
    let depth = 0;
    for (let i = 0; i < first.length; i++) {
      if (items.every(e => e.tokens[i] === first[i])) depth = i + 1;
      else break;
    }
    return depth;
  }

  // Reconstruct a display label from the first N tokens of an entry,
  // using the original separator characters from the bare name.
  function prefixLabel(entry, depth) {
    if (depth === 0) return entry.bare;
    // Walk the bare string collecting the first `depth` tokens worth of chars
    let count = 0;
    let i = 0;
    while (i < entry.bare.length && count < depth) {
      if (entry.bare[i] === '-' || entry.bare[i] === '_') count++;
      i++;
    }
    // If we consumed all tokens without hitting a separator, include the rest
    return entry.bare.slice(0, count < depth ? entry.bare.length : i - 1);
  }

  // Suffix display label after removing a shared prefix of `depth` tokens.
  function suffixLabel(entry, depth) {
    const full = entry.bare;
    let count = 0;
    let i = 0;
    while (i < full.length && count < depth) {
      if (full[i] === '-' || full[i] === '_') count++;
      i++;
    }
    const suffix = count < depth ? '' : full.slice(i);
    return suffix || '(base)'; // base model of this series
  }

  // Cluster a list of entries into groups by longest shared prefix.
  // Returns array of { prefixDepth, label, items }.
  function clusterByPrefix(items) {
    if (items.length === 0) return [];

    // Build a map from prefix-at-depth-1 → items sharing that prefix
    const byFirst = new Map();
    for (const e of items) {
      const key = e.tokens[0] || '';
      if (!byFirst.has(key)) byFirst.set(key, []);
      byFirst.get(key).push(e);
    }

    const clusters = [];
    for (const [, group] of byFirst) {
      if (group.length === 1) {
        // Singleton — no grouping
        clusters.push({ prefixDepth: 0, label: group[0].bare, items: group });
      } else {
        // Find the longest prefix depth for this group
        const depth = longestSharedPrefix(group);
        const label = prefixLabel(group[0], depth);
        clusters.push({ prefixDepth: depth, label, items: group });
      }
    }

    // Preserve original model order within each cluster
    const orderMap = new Map(items.map((e, i) => [e.model.model_id, i]));
    clusters.sort((a, b) => {
      const ai = orderMap.get(a.items[0].model.model_id) ?? 0;
      const bi = orderMap.get(b.items[0].model.model_id) ?? 0;
      return ai - bi;
    });

    return clusters;
  }

  const topClusters = clusterByPrefix(entries);
  const result = [];

  for (const cluster of topClusters) {
    if (cluster.prefixDepth === 0) {
      // Singleton flat row
      result.push({ type: 'row', model: cluster.items[0].model, variantLabel: cluster.label });
    } else {
      // Group: check for sub-groups within
      const groupItems = cluster.items;
      const subClusters = clusterByPrefix(
        groupItems.map(e => ({
          model: e.model,
          bare: suffixLabel(e, cluster.prefixDepth),
          tokens: tokenizeName(suffixLabel(e, cluster.prefixDepth)),
        }))
      );

      const children = [];
      for (const sub of subClusters) {
        if (sub.prefixDepth === 0) {
          children.push({ type: 'row', model: sub.items[0].model, variantLabel: sub.label });
        } else {
          const subChildren = sub.items.map(se => ({
            type: 'row',
            model: se.model,
            variantLabel: suffixLabel(se, sub.prefixDepth),
          }));
          children.push({ type: 'subgroup', label: sub.label, children: subChildren });
        }
      }

      result.push({ type: 'group', label: cluster.label, children });
    }
  }

  return result;
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

/* ── Licenses section ────────────────────────────────────────────────────────*/

let allLicenses = [];
let activeLicenseSlug = null;

async function loadLicenseList() {
  const res = await fetch('/api/licenses');
  if (!res.ok) throw new Error(`Failed to fetch licenses: ${res.status}`);
  allLicenses = await res.json();
  renderLicenseList(allLicenses);
}

function renderLicenseList(licenses) {
  const ul = document.getElementById('license-list');
  if (!ul) return;
  ul.innerHTML = '';
  licenses.forEach(lic => {
    const li = document.createElement('li');
    li.dataset.slug = lic.slug;
    const label = lic.display_name && lic.display_name !== lic.slug
      ? `${escHtml(lic.display_name)} <span style="color:var(--gray-400);font-size:11px">${escHtml(lic.slug)}</span>`
      : escHtml(lic.slug);
    li.innerHTML = label;
    li.title = `${lic.model_count} model${lic.model_count !== 1 ? 's' : ''}`;
    if (lic.slug === activeLicenseSlug) li.classList.add('selected');
    li.addEventListener('click', () => selectLicense(lic.slug, li));
    ul.appendChild(li);
  });
}

function selectLicense(slug, liEl) {
  document.querySelectorAll('#license-list li').forEach(el => el.classList.remove('selected'));
  if (liEl) liEl.classList.add('selected');
  activeLicenseSlug = slug;
  loadLicenseDetail(slug);
}

async function loadLicenseDetail(slug) {
  const panel = document.getElementById('license-detail');
  if (!panel) return;
  panel.innerHTML = '<div class="detail-empty">Loading…</div>';
  const res = await fetch(`/api/licenses/${encodeURIComponent(slug)}`);
  if (!res.ok) {
    panel.innerHTML = `<div class="detail-empty">Error loading license (${res.status})</div>`;
    return;
  }
  const data = await res.json();
  renderLicenseDetail(panel, data);
}

function renderLicenseDetail(panel, d) {
  const familyBadge = d.family
    ? `<span class="license-badge license-badge--${escHtml(d.family)}">${escHtml(d.family)}</span>`
    : '';
  const osiBadge = d.is_osi_approved === true
    ? '<span class="license-badge license-badge--osi">OSI Approved</span>'
    : d.is_osi_approved === false
      ? '<span class="license-badge license-badge--not-osi">Not OSI</span>'
      : '';
  const sourceLink = d.source_url
    ? `<a href="${escHtml(d.source_url)}" target="_blank" rel="noopener" class="license-source-link">View source ↗</a>`
    : '';

  const boolCell = v =>
    v === true ? '<span style="color:#16a34a;font-weight:600">Yes</span>'
    : v === false ? '<span style="color:#dc2626">No</span>'
    : '—';

  const companiesHtml = (d.companies || []).map(co => `
    <div class="license-company-group">
      <span class="license-company-name">${escHtml(co.company_name)}</span>
      <span class="license-company-count">${co.model_count} model${co.model_count !== 1 ? 's' : ''}</span>
    </div>
  `).join('');

  const textBlock = d.license_text
    ? `<div class="section-title">License Text ${sourceLink}</div>
       <pre class="license-text-pre">${escHtml(d.license_text)}</pre>`
    : `<div class="section-title">License Text ${sourceLink}</div>
       <div class="license-text-missing">License text not yet fetched — run huggingface/fetch_license_texts.py</div>`;

  panel.innerHTML = `
    <div class="company-header">
      <div class="company-name">${escHtml(d.display_name || d.slug)}</div>
      <div class="company-handle">${escHtml(d.slug)}</div>
      <div class="license-badges">${familyBadge}${osiBadge}</div>
    </div>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Commercial Use</div>
        <div class="stat-value small">${boolCell(d.allows_commercial_use)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Derivatives</div>
        <div class="stat-value small">${boolCell(d.allows_derivatives)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Attribution</div>
        <div class="stat-value small">${boolCell(d.requires_attribution)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Share-Alike</div>
        <div class="stat-value small">${boolCell(d.requires_share_alike)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Models Using</div>
        <div class="stat-value">${(d.companies || []).reduce((s, c) => s + c.model_count, 0)}</div>
      </div>
    </div>

    ${d.notes ? `<div class="section-title">Notes</div><p class="license-notes">${escHtml(d.notes)}</p>` : ''}

    <div class="section-title">Companies Using This License</div>
    <div class="license-companies-list">
      ${companiesHtml || '<div class="license-text-missing">No models recorded.</div>'}
    </div>

    ${textBlock}
  `;
}

document.getElementById('license-search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  const filtered = allLicenses.filter(l =>
    (l.slug || '').toLowerCase().includes(q) ||
    (l.display_name || '').toLowerCase().includes(q)
  );
  renderLicenseList(filtered);
});


/* ── Status badge ───────────────────────────────────────────────────────────*/
async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();
    const badge = document.getElementById('last-updated-badge');
    if (!badge) return;
    if (!d.last_collected_at) {
      badge.textContent = 'Data: Jan 6 2026';
      return;
    }
    const dt = new Date(d.last_collected_at);
    const fmt = dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    badge.textContent = `Last updated: ${fmt}`;
    badge.title = `${d.new_models_added ?? 0} new models added · ${d.total_models_in_db} total`;
  } catch (_) { /* non-fatal */ }
}

/* ── Home stats ─────────────────────────────────────────────────────────────*/

async function loadHomeStats() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const d = await res.json();
    const updatedEl = document.getElementById('home-updated');
    if (!updatedEl) return;
    if (d.last_collected_at) {
      const dt = new Date(d.last_collected_at);
      updatedEl.textContent = `Last updated: ${dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
    } else {
      updatedEl.textContent = 'Data: Jan 6, 2026';
    }
  } catch (_) {}
}

/* ── Init ───────────────────────────────────────────────────────────────────*/
loadStatus();
loadHomeStats();
loadCompanyList().catch(err => {
  console.error('Failed to load companies:', err);
  const ul = document.getElementById('company-list');
  if (ul) ul.innerHTML = '<li style="padding:.75rem 1rem;color:#ef4444">Failed to load</li>';
});
loadLicenseList().catch(err => {
  console.error('Failed to load licenses:', err);
  const ul = document.getElementById('license-list');
  if (ul) ul.innerHTML = '<li style="padding:.75rem 1rem;color:#ef4444">Failed to load</li>';
});

let analysisNavReady = false;
let historicalNavReady = false;

function setupAnalysisSection() {
  const section = document.getElementById('section-analysis');
  if (!section || !section.classList.contains('active')) return;

  // Wire sub-tab buttons (idempotent via onclick assignment)
  section.querySelectorAll('.analysis-subtab').forEach(btn => {
    btn.onclick = () => switchAnalysisSubtab(btn.dataset.subtab);
  });

  // Init whichever sub-panel is currently active
  const activeSubtab = section.querySelector('.analysis-subtab.active')?.dataset.subtab || 'snapshot';
  if (activeSubtab === 'snapshot' && !analysisNavReady) {
    setupAnalysisNavigation();
    analysisNavReady = true;
  }
  if (activeSubtab === 'historical' && !historicalNavReady) {
    setupHistoricalNavigation();
  }
}

function switchAnalysisSubtab(name) {
  document.querySelectorAll('.analysis-subtab').forEach(b => b.classList.toggle('active', b.dataset.subtab === name));
  document.querySelectorAll('.analysis-subpanel').forEach(p => p.classList.toggle('active', p.id === `analysis-subpanel-${name}`));

  if (name === 'snapshot' && !analysisNavReady) {
    setupAnalysisNavigation();
    analysisNavReady = true;
  }
  if (name === 'historical' && !historicalNavReady) {
    setupHistoricalNavigation();
  }
}

setupAnalysisSection();
window.addEventListener('hashchange', () => setTimeout(setupAnalysisSection, 50));


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

/* ── Models section ──────────────────────────────────────────────────────────*/

let modelsLoaded = false;
let currentModelsPage = 0;
const MODELS_PAGE_SIZE = 50;

async function loadModelsFilters() {
  const [filters, companies, licenses] = await Promise.all([
    fetch('/api/models/filters').then(r => r.json()),
    fetch('/api/companies').then(r => r.json()),
    fetch('/api/licenses').then(r => r.json()),
  ]);

  const companySelect = document.getElementById('models-filter-company');
  companies.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.display_name;
    companySelect.appendChild(opt);
  });

  const licenseSelect = document.getElementById('models-filter-license');
  licenses.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l.slug;
    opt.textContent = l.display_name && l.display_name !== l.slug
      ? `${l.display_name} (${l.slug})`
      : l.slug;
    licenseSelect.appendChild(opt);
  });

  const countrySelect = document.getElementById('models-filter-country');
  filters.countries.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    countrySelect.appendChild(opt);
  });

  const modalitySelect = document.getElementById('models-filter-modality');
  filters.modalities.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modalitySelect.appendChild(opt);
  });

  const typologySelect = document.getElementById('models-filter-typology');
  (filters.typology_types || []).forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    typologySelect.appendChild(opt);
  });

  // Filter change → reload list
  ['models-filter-company', 'models-filter-license', 'models-filter-country',
   'models-filter-source', 'models-filter-modality', 'models-filter-typology'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => loadModelsList(0));
  });

  document.getElementById('models-clear-filters').addEventListener('click', () => {
    ['models-filter-company', 'models-filter-license', 'models-filter-country',
     'models-filter-source', 'models-filter-modality', 'models-filter-typology'].forEach(id => {
      document.getElementById(id).value = '';
    });
    loadModelsList(0);
  });

  document.getElementById('models-back-btn').addEventListener('click', () => {
    document.getElementById('models-table-view').style.display = '';
    document.getElementById('models-detail-view').style.display = 'none';
    document.querySelectorAll('#models-tbody tr.selected').forEach(r => r.classList.remove('selected'));
  });
}

async function loadModelsList(page = 0) {
  currentModelsPage = page;
  const params = new URLSearchParams({ limit: MODELS_PAGE_SIZE, offset: page * MODELS_PAGE_SIZE });

  const company  = document.getElementById('models-filter-company').value;
  const license  = document.getElementById('models-filter-license').value;
  const country  = document.getElementById('models-filter-country').value;
  const source   = document.getElementById('models-filter-source').value;
  const modality = document.getElementById('models-filter-modality').value;
  const typology = document.getElementById('models-filter-typology').value;

  if (company)  params.set('company_id',    company);
  if (license)  params.set('license_slug',  license);
  if (country)  params.set('country_hq',    country);
  if (source)   params.set('data_source',   source);
  if (modality) params.set('modality',      modality);
  if (typology) params.set('typology_type', typology);

  // Show table, hide detail
  document.getElementById('models-table-view').style.display = '';
  document.getElementById('models-detail-view').style.display = 'none';

  const tbody = document.getElementById('models-tbody');
  tbody.innerHTML = '<tr><td colspan="9" class="models-loading">Loading…</td></tr>';

  try {
    const data = await fetch(`/api/models?${params}`).then(r => r.json());
    renderModelsTable(data.models, data.total, page);
    document.getElementById('models-result-count').textContent =
      `${data.total.toLocaleString()} model${data.total !== 1 ? 's' : ''}`;
  } catch (err) {
    console.error('Failed to load models:', err);
    tbody.innerHTML = '<tr><td colspan="9" class="models-loading">Failed to load models</td></tr>';
  }
}

function renderModelsTable(models, total, page) {
  const tbody = document.getElementById('models-tbody');
  if (!models || models.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="models-empty">No models match current filters</td></tr>';
    document.getElementById('models-pagination').innerHTML = '';
    return;
  }

  tbody.innerHTML = '';
  models.forEach(m => {
    const tr = document.createElement('tr');
    tr.dataset.modelId = m.model_id;

    const params    = m.num_parameters != null ? Number(m.num_parameters).toFixed(1) : '—';
    const date      = m.release_date   ? String(m.release_date).slice(0, 10) : '—';
    const downloads = m.downloads      != null ? m.downloads.toLocaleString() : '—';
    const likes     = m.likes          != null ? m.likes.toLocaleString()     : '—';

    tr.innerHTML = `
      <td class="model-id-cell" title="${escHtml(m.model_id)}">${escHtml(m.model_id)}</td>
      <td>${escHtml(m.company_name || '—')}</td>
      <td><span class="models-license-tag">${escHtml(m.license_slug || '—')}</span></td>
      <td>${escHtml(m.modality || '—')}</td>
      <td>${m.typology_type ? `<span class="typology-type-badge typology-${escHtml(m.typology_type)}">${escHtml(m.typology_type)}</span>` : '—'}</td>
      <td class="num-cell">${params}</td>
      <td class="num-cell">${date}</td>
      <td class="num-cell">${downloads}</td>
      <td class="num-cell">${likes}</td>
    `;

    tr.addEventListener('click', () => {
      document.querySelectorAll('#models-tbody tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      loadModelDetail(m.model_id);
    });

    tbody.appendChild(tr);
  });

  // Pagination controls
  const totalPages  = Math.ceil(total / MODELS_PAGE_SIZE);
  const paginationEl = document.getElementById('models-pagination');
  paginationEl.innerHTML = `
    <button id="models-prev-btn" ${page === 0 ? 'disabled' : ''}>← Prev</button>
    <button id="models-next-btn" ${page >= totalPages - 1 ? 'disabled' : ''}>Next →</button>
    <span class="models-pagination-info">Page ${page + 1} of ${totalPages} · ${total.toLocaleString()} models</span>
  `;
  document.getElementById('models-prev-btn').addEventListener('click', () => loadModelsList(page - 1));
  document.getElementById('models-next-btn').addEventListener('click', () => loadModelsList(page + 1));
}

async function loadModelDetail(modelId) {
  document.getElementById('models-table-view').style.display = 'none';
  document.getElementById('models-detail-view').style.display = '';

  const contentEl = document.getElementById('models-detail-content');
  contentEl.innerHTML = '<div class="detail-empty">Loading…</div>';

  try {
    const res = await fetch(`/api/models/${modelId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderModelDetail(contentEl, await res.json());
  } catch (err) {
    console.error('Failed to load model detail:', err);
    contentEl.innerHTML = '<div class="detail-empty">Failed to load model details</div>';
  }
}

function renderModelDetail(container, d) {
  const meta = d.metadata || {};

  const urlHtml = d.url
    ? `<a href="${escHtml(d.url)}" target="_blank" rel="noopener" class="license-source-link">${escHtml(d.url)} ↗</a>`
    : '';

  const licenseHtml = d.license_slug
    ? `<a href="#licenses" class="license-link models-detail-license" data-slug="${escHtml(d.license_slug)}">${escHtml(d.license_slug)}</a>`
    : '—';

  const params  = meta.num_parameters != null ? `${Number(meta.num_parameters).toFixed(1)}B` : '—';
  const date    = d.release_date      ? String(d.release_date).slice(0, 10)           : '—';
  const lastMod = meta.last_modified  ? String(meta.last_modified).slice(0, 10)       : '—';

  const tagsHtml = (meta.tags || []).slice(0, 20)
    .map(t => `<span class="model-tag">${escHtml(t)}</span>`).join('');
  const archHtml = (meta.architectures || [])
    .map(a => `<span class="model-tag">${escHtml(a)}</span>`).join('');

  container.innerHTML = `
    <div class="company-header">
      <div class="company-name">${escHtml(d.display_name || d.model_id)}</div>
      <div class="company-handle">${escHtml(d.model_id)}</div>
      ${urlHtml}
    </div>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Company</div>
        <div class="stat-value small">${escHtml(d.company_name || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Country</div>
        <div class="stat-value small">${escHtml(d.country_hq || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">License</div>
        <div class="stat-value small">${licenseHtml}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Source</div>
        <div class="stat-value small">${escHtml(d.data_source || '—')}</div>
      </div>
    </div>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Parameters</div>
        <div class="stat-value small">${params}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Modality</div>
        <div class="stat-value small">${escHtml(meta.modality || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Released</div>
        <div class="stat-value small">${date}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Last Modified</div>
        <div class="stat-value small">${lastMod}</div>
      </div>
    </div>

    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Downloads</div>
        <div class="stat-value">${d.downloads != null ? d.downloads.toLocaleString() : '—'}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Likes</div>
        <div class="stat-value">${d.likes != null ? d.likes.toLocaleString() : '—'}</div>
      </div>
      ${d.context_tokens ? `
      <div class="stat-card">
        <div class="stat-label">Context Tokens</div>
        <div class="stat-value">${d.context_tokens.toLocaleString()}</div>
      </div>` : ''}
    </div>

    ${meta.pipeline_tag ? `
    <div class="section-title" style="margin-top:1rem">Pipeline</div>
    <p class="model-meta-text">${escHtml(meta.pipeline_tag)}</p>
    ` : ''}

    ${meta.library_name || meta.model_type ? `
    <div class="section-title">Framework / Model Type</div>
    <p class="model-meta-text">${[meta.library_name, meta.model_type].filter(Boolean).map(escHtml).join(' · ')}</p>
    ` : ''}

    ${archHtml ? `
    <div class="section-title">Architectures</div>
    <div class="model-tags">${archHtml}</div>
    ` : ''}

    ${tagsHtml ? `
    <div class="section-title" style="margin-top:1rem">Tags</div>
    <div class="model-tags">${tagsHtml}</div>
    ` : ''}

    ${meta.typology_type ? `
    <div class="section-title" style="margin-top:1rem">Typology Classification</div>
    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Type</div>
        <div class="stat-value small"><span class="typology-type-badge typology-${escHtml(meta.typology_type)}">${escHtml(meta.typology_type)}</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Confidence</div>
        <div class="stat-value small">${escHtml(meta.typology_confidence || '—')}</div>
      </div>
    </div>
    ${meta.typology_tags && meta.typology_tags.length ? `
    <div class="model-tags" style="margin-top:.5rem">
      ${meta.typology_tags.map(t => `<span class="model-tag typology-tag">${escHtml(t)}</span>`).join('')}
    </div>` : ''}
    ${meta.typology_rationale ? `
    <div class="section-title" style="margin-top:.75rem">Classification Rationale</div>
    <p class="model-meta-text">${escHtml(meta.typology_rationale)}</p>
    ` : ''}
    ` : ''}
  `;

  // Wire license link click-through to licenses section
  container.querySelectorAll('.models-detail-license').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const slug = a.dataset.slug;
      location.hash = 'licenses';
      setTimeout(() => {
        const li = document.querySelector(`#license-list li[data-slug="${CSS.escape(slug)}"]`);
        if (li) {
          selectLicense(slug, li);
          li.scrollIntoView({ block: 'nearest' });
        } else {
          loadLicenseDetail(slug);
        }
      }, 80);
    });
  });
}

const setupModels = () => {
  const section = document.getElementById('section-models');
  if (section && section.classList.contains('active') && !modelsLoaded) {
    modelsLoaded = true;
    loadModelsFilters()
      .then(() => loadModelsList(0))
      .catch(err => console.error('Failed to initialize models section:', err));
  }
};

setupModels();
window.addEventListener('hashchange', () => setTimeout(setupModels, 50));


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


/* ── Map section ─────────────────────────────────────────────────────────────
 * Interactive world map of AI company headquarters and affiliate offices.
 * Libraries: Leaflet 1.9.4, topojson-client 3 (both via CDN).
 * Arcs: pure L.Polyline with parabolic intermediate points (no plugin needed).
 * HQ markers: L.divIcon diamond shape, visually distinct from affiliate circles.
 */

let mapLoaded = false;
let leafletMap = null;

// Fixed color per company_id
const MAP_COLORS = {
  14: '#2563eb',  // OpenAI
  24: '#7c3aed',  // Anthropic
   8: '#059669',  // Google DeepMind
  22: '#dc2626',  // xAI
   9: '#0284c7',  // Meta AI (FAIR)
   5: '#d97706',  // Cohere
  11: '#db2777',  // Mistral AI
   4: '#65a30d',  // Baidu
   6: '#0891b2',  // DeepSeek
  17: '#ea580c',  // Qwen
};
const _FALLBACK_PALETTE = ['#6366f1','#14b8a6','#f59e0b','#84cc16','#a855f7'];
let _fallbackIdx = 0;

function getCompanyColor(company_id) {
  if (!MAP_COLORS[company_id]) {
    MAP_COLORS[company_id] = _FALLBACK_PALETTE[_fallbackIdx++ % _FALLBACK_PALETTE.length];
  }
  return MAP_COLORS[company_id];
}

// ISO numeric → country name (for world-atlas choropleth matching)
const COUNTRY_ISO_NUM = {
  'United States': '840', 'USA': '840',
  'United Kingdom': '826',
  'Canada':        '124',
  'France':        '250',
  'Germany':       '276',
  'China':         '156',
  'Japan':         '392',
  'South Korea':   '410',
  'Singapore':     '702',
  'Belgium':       '056',
  'Ireland':       '372',
  'Switzerland':   '756',
  'India':         '356',
  'Israel':        '376',
  'Australia':     '036',
  'Taiwan':        '158',
};

// ── Arc helper ───────────────────────────────────────────────────────────────
// Parabolic arc using L.Polyline — no plugin needed.
// Bump height is capped at 15° latitude so transoceanic routes stay on the map.
function makeArc(from, to, color, confirmed) {
  const [lat1, lng1] = from;
  const [lat2, lng2] = to;
  const NUM_PTS = 80;
  const dist = Math.sqrt((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2);
  const bump = Math.min(dist * 0.28, 15);  // cap: never exceeds 15° of latitude

  const pts = [];
  for (let i = 0; i <= NUM_PTS; i++) {
    const t = i / NUM_PTS;
    pts.push([
      lat1 + (lat2 - lat1) * t + Math.sin(Math.PI * t) * bump,
      lng1 + (lng2 - lng1) * t,
    ]);
  }
  return L.polyline(pts, {
    color,
    weight:      confirmed ? 2 : 1,
    opacity:     confirmed ? 0.7 : 0.3,
    dashArray:   confirmed ? null : '6 5',
    smoothFactor: 1,
  });
}

// ── HQ diamond icon ──────────────────────────────────────────────────────────
// A rotated square gives a clear diamond shape, distinct from circle affiliates.
function makeHQIcon(color) {
  const S = 14;  // diamond size (px, before rotation)
  return L.divIcon({
    className: '',
    html: `<div style="
      width:${S}px;height:${S}px;
      background:${color};
      border:2px solid #fff;
      transform:rotate(45deg);
      box-shadow:0 0 0 1.5px ${color},0 1px 6px rgba(0,0,0,.5);
    "></div>`,
    iconSize:    [S, S],
    iconAnchor:  [S / 2, S / 2],
    popupAnchor: [0, -S / 2 - 4],
  });
}

// ── Main init ────────────────────────────────────────────────────────────────
async function initMap() {
  const res = await fetch('/api/research/map');
  if (!res.ok) { console.error('Failed to load map data'); return; }
  const rows = await res.json();

  // Group rows by company
  const companies = {};
  rows.forEach(r => {
    if (!companies[r.company_id])
      companies[r.company_id] = { name: r.company_name, hq: null, affiliates: [] };
    const entry = { ...r, latlng: [r.latitude, r.longitude] };
    if (r.finding_type === 'headquarters') companies[r.company_id].hq = entry;
    else                                   companies[r.company_id].affiliates.push(entry);
  });

  // Country score for choropleth: HQ=3, confirmed affiliate=1
  const countryScores = {};
  rows.forEach(r => {
    if (!countryScores[r.country]) countryScores[r.country] = 0;
    countryScores[r.country] += r.finding_type === 'headquarters' ? 3
                              : r.confidence === 'confirmed'      ? 1 : 0;
  });
  const maxScore = Math.max(...Object.values(countryScores), 1);

  // Reverse map for choropleth: ISO numeric id → country name
  const isoToCountry = {};
  Object.entries(COUNTRY_ISO_NUM).forEach(([name, iso]) => { isoToCountry[iso] = name; });

  // Init Leaflet
  leafletMap = L.map('map', { center: [25, 10], zoom: 2, minZoom: 2 });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 18,
  }).addTo(leafletMap);

  // Country choropleth (world-atlas TopoJSON)
  try {
    const topo = await (await fetch(
      'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'
    )).json();
    const geojson = topojson.feature(topo, topo.objects.countries);
    L.geoJSON(geojson, {
      style: f => {
        const iso   = String(f.id).padStart(3, '0');
        const name  = isoToCountry[iso];
        const score = name ? (countryScores[name] || 0) : 0;
        return {
          fillColor:   '#2563eb',
          fillOpacity: score > 0 ? Math.min(0.55, 0.1 + (score / maxScore) * 0.5) : 0,
          color:       '#94a3b8',
          weight:      0.4,
          opacity:     0.5,
        };
      },
    }).addTo(leafletMap);
  } catch(e) { console.warn('World boundaries failed to load:', e); }

  // ── Build per-company layer groups ──────────────────────────────────────────
  // Each "item" tracks a marker+arc pair for filter management.
  const companyGroups = {};  // cid → L.layerGroup
  // items: [{marker, arc|null, companyId, country}]
  const items = [];

  Object.entries(companies).forEach(([cid, comp]) => {
    const cidNum = Number(cid);
    const color  = getCompanyColor(cidNum);
    const group  = L.layerGroup();
    companyGroups[cid] = group;

    // HQ: diamond icon
    if (comp.hq) {
      const hqMarker = L.marker(comp.hq.latlng, { icon: makeHQIcon(color) })
        .bindPopup(`<b>${comp.name}</b><br><b>HQ</b>: ${comp.hq.city}, ${comp.hq.country}`);
      hqMarker._finding = comp.hq;
      group.addLayer(hqMarker);
      items.push({ marker: hqMarker, arc: null, companyId: cidNum, country: comp.hq.country });
    }

    // Affiliates: filled circles + arcs from HQ
    comp.affiliates.forEach(aff => {
      const confirmed = aff.confidence === 'confirmed';
      const dot = L.circleMarker(aff.latlng, {
        radius:      confirmed ? 5 : 4,
        color:       '#fff',
        weight:      1.5,
        fillColor:   color,
        fillOpacity: confirmed ? 0.85 : 0.4,
      }).bindPopup(
        `<b>${comp.name}</b><br>Office: ${aff.city}, ${aff.country}`
        + (confirmed ? '' : '<br><i style="color:#9ca3af">unconfirmed</i>')
      );
      dot._finding = aff;
      group.addLayer(dot);

      let arc = null;
      if (comp.hq) {
        arc = makeArc(comp.hq.latlng, aff.latlng, color, confirmed);
        arc._finding = aff;
        group.addLayer(arc);
      }
      items.push({ marker: dot, arc, companyId: cidNum, country: aff.country });
    });

    group.addTo(leafletMap);
  });

  // ── Filter state & apply ────────────────────────────────────────────────────
  const activeCompanies = new Set(Object.keys(companies).map(Number));
  const activeCountries  = new Set(rows.map(r => r.country));

  function applyFilters() {
    items.forEach(({ marker, arc, companyId, country }) => {
      const group   = companyGroups[companyId];
      const visible = activeCompanies.has(companyId) && activeCountries.has(country);

      // Marker
      if (visible && !group.hasLayer(marker)) group.addLayer(marker);
      if (!visible && group.hasLayer(marker))  group.removeLayer(marker);

      // Arc (polyline)
      if (arc) {
        if (visible && !group.hasLayer(arc)) group.addLayer(arc);
        if (!visible && group.hasLayer(arc))  group.removeLayer(arc);
      }
    });

    // Show/hide entire company group
    Object.entries(companyGroups).forEach(([cid, group]) => {
      const on = activeCompanies.has(Number(cid));
      if (on  && !leafletMap.hasLayer(group)) group.addTo(leafletMap);
      if (!on &&  leafletMap.hasLayer(group)) leafletMap.removeLayer(group);
    });
  }

  // ── Sidebar ─────────────────────────────────────────────────────────────────

  // Company toggles (alphabetical, with diamond swatch for HQ shape cue)
  const companyToggleEl = document.getElementById('map-company-toggles');
  Object.entries(companies)
    .sort((a, b) => a[1].name.localeCompare(b[1].name))
    .forEach(([cid, comp]) => {
      const color = getCompanyColor(Number(cid));
      const label = document.createElement('label');
      label.className = 'map-toggle-item';
      label.innerHTML = `
        <input type="checkbox" checked data-cid="${cid}" />
        <span class="map-swatch" style="background:${color};border-radius:2px;transform:rotate(45deg)"></span>
        <span>${comp.name}</span>`;
      label.querySelector('input').addEventListener('change', e => {
        const id = Number(e.target.dataset.cid);
        e.target.checked ? activeCompanies.add(id) : activeCompanies.delete(id);
        applyFilters();
      });
      companyToggleEl.appendChild(label);
    });

  // Country toggles (alphabetical)
  const countryToggleEl = document.getElementById('map-country-toggles');
  [...new Set(rows.map(r => r.country))].sort().forEach(country => {
    const label = document.createElement('label');
    label.className = 'map-toggle-item';
    label.innerHTML = `
      <input type="checkbox" checked data-country="${country}" />
      <span>${country}</span>`;
    label.querySelector('input').addEventListener('change', e => {
      const c = e.target.dataset.country;
      e.target.checked ? activeCountries.add(c) : activeCountries.delete(c);
      applyFilters();
    });
    countryToggleEl.appendChild(label);
  });

  // Recalculate map size (it was hidden when Leaflet initialised)
  setTimeout(() => leafletMap.invalidateSize(), 100);
}

const setupMapSection = () => {
  const section = document.getElementById('section-map');
  if (section && section.classList.contains('active') && !mapLoaded) {
    mapLoaded = true;
    initMap().catch(console.error);
  }
};

setupMapSection();
window.addEventListener('hashchange', () => setTimeout(setupMapSection, 50));

