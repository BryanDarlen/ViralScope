const PRESETS = {
  "YouTube breakout": {
    platform: "YouTube",
    niche: "productivity",
    media_type: "short_video",
    caption: "I tried studying 12 hours using AI tools",
    hashtags: "#study #aitools #productivity",
    account_follower_count: 25000,
    account_age_days: 730,
    early_window_hours: 2,
    early_views: 18000,
    early_likes: 2100,
    early_comments: 320,
    early_shares: 450,
    post_time: "20:00"
  },
  "TikTok coder": {
    platform: "TikTok",
    niche: "coding",
    media_type: "short_video",
    caption: "POV: your code finally works at 3AM",
    hashtags: "#coding #programming #studentlife",
    account_follower_count: 3000,
    account_age_days: 365,
    early_window_hours: 1,
    early_views: 9500,
    early_likes: 1400,
    early_comments: 180,
    early_shares: 300,
    post_time: "22:00"
  },
  "Reddit slow start": {
    platform: "Reddit",
    niche: "productivity",
    media_type: "text",
    caption: "My thoughts on productivity",
    hashtags: "",
    account_follower_count: 60,
    account_age_days: 20,
    early_window_hours: 2,
    early_views: 40,
    early_likes: 4,
    early_comments: 0,
    early_shares: 0,
    post_time: "03:00"
  }
};

const state = {
  bootstrap: null,
  comparison: null
};

document.addEventListener("DOMContentLoaded", () => {
  bindNavigation();
  bindPredictionForm();
  bindYouTubeLiveForm();
  bindTrainingForms();
  seedPresetButtons();
  setDefaultDate();
  bootstrap();
});

function bindNavigation() {
  document.querySelectorAll(".nav-link").forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`view-${view}`).classList.add("active");
    });
  });
}

function seedPresetButtons() {
  const presetRow = document.getElementById("preset-row");
  Object.keys(PRESETS).forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.textContent = name;
    button.addEventListener("click", () => applyPreset(name));
    presetRow.appendChild(button);
  });
  applyPreset("YouTube breakout");
}

function setDefaultDate() {
  const dateInput = document.getElementById("post_date");
  if (!dateInput.value) {
    dateInput.value = new Date().toISOString().slice(0, 10);
  }
}

async function bootstrap() {
  try {
    const payload = await fetchJSON("/api/bootstrap");
    state.bootstrap = payload;
    hydrateConfig(payload.config);
    renderDashboard(payload);
    showFlash(payload.message || "", false, !payload.message);
  } catch (error) {
    showFlash(error.message || "Failed to load dashboard.", true);
  }
}

function hydrateConfig(config) {
  populateSelect("platform", config.supported_platforms);
  populateSelect("niche", config.supported_niches);
  document.getElementById("demo_rows").value = config.current_demo_rows || 1800;
}

function populateSelect(id, values) {
  const select = document.getElementById(id);
  const current = select.value;
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  if (current && values.includes(current)) {
    select.value = current;
  }
}

function renderDashboard(payload) {
  renderSidebar(payload.model);
  renderHero(payload);
  renderDatasetSummary(payload.dataset_summary);
  renderPlatformMix(payload.platform_mix);
  renderViralityByPlatform(payload.virality_by_platform);
  renderEvaluation(payload.evaluation, payload.model);
  renderFeatureImportance(payload.model.feature_importance || []);
  renderWatchlist(payload.trend_watchlist);
  renderPreviewTable(payload.dataset_preview);
}

function renderSidebar(model) {
  document.getElementById("sidebar-model-name").textContent = model.model_name;
  document.getElementById("sidebar-model-source").textContent = model.source.replaceAll("_", " ");
  document.getElementById("sidebar-rows").textContent = formatInt(model.rows);
  document.getElementById("sidebar-positive-rate").textContent = `${(model.positive_rate * 100).toFixed(1)}%`;
  document.getElementById("predict-model-name").textContent = model.model_name;
  document.getElementById("predict-data-mode").textContent = model.source.replaceAll("_", " ");
}

function renderHero(payload) {
  const chips = document.getElementById("watchlist-chips");
  chips.innerHTML = "";
  payload.trend_watchlist.slice(0, 4).forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = item.niche;
    chips.appendChild(chip);
  });

  const heroStats = document.getElementById("hero-stats");
  heroStats.innerHTML = "";
  const stats = [
    {
      label: "Active model",
      value: payload.model.model_name,
      copy: "Ready for prediction"
    },
    {
      label: "Viral share",
      value: `${(payload.model.positive_rate * 100).toFixed(1)}%`,
      copy: "Rare-positive benchmark rate"
    },
    {
      label: "Rows tracked",
      value: formatInt(payload.model.rows),
      copy: "Current active training context"
    },
    {
      label: "Data source",
      value: payload.model.source.replaceAll("_", " "),
      copy: "Local-first model workflow"
    }
  ];

  stats.forEach((item) => {
    const card = document.createElement("div");
    card.className = "hero-stat";
    card.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong><small>${item.copy}</small>`;
    heroStats.appendChild(card);
  });
}

function renderDatasetSummary(summary) {
  const grid = document.getElementById("dataset-summary-grid");
  const healthGrid = document.getElementById("health-metrics");
  const cards = [
    { label: "Rows", value: formatInt(summary.rows), copy: "Posts in the current benchmark set" },
    { label: "Viral rate", value: `${(summary.viral_rate * 100).toFixed(1)}%`, copy: "Share of positive examples" },
    { label: "Platforms", value: `${summary.platform_count}`, copy: "Cross-network coverage" },
    { label: "Median early views", value: formatInt(summary.median_early_views), copy: "Benchmark demand snapshot" }
  ];
  grid.innerHTML = cards.map(metricCard).join("");

  const metrics = state.bootstrap?.model?.metrics || {};
  const validation = state.bootstrap?.model?.validation || {};
  const healthCards = [
    { label: "ROC AUC", value: metricValue(metrics.roc_auc), copy: "Ranking quality across the decision surface" },
    { label: "Average precision", value: metricValue(metrics.average_precision), copy: "Rare-positive performance summary" },
    { label: "F1 score", value: metricValue(metrics.f1), copy: "Balance of catch rate and precision" },
    { label: "Brier score", value: metricValue(metrics.brier_score), copy: "Probability calibration quality" },
    { label: "Validation", value: validationLabel(validation), copy: "How the holdout set was chosen" },
    { label: "Holdout window", value: validationWindow(validation), copy: "Later posts reserved for credibility checks" }
  ];
  healthGrid.innerHTML = healthCards.map(metricCard).join("");
}

function metricCard(item) {
  return `<div class="metric-card"><span>${item.label}</span><strong>${item.value}</strong><small>${item.copy}</small></div>`;
}

function renderPlatformMix(rows) {
  document.getElementById("platform-mix-bars").innerHTML = renderBarList(rows, "posts", "platform", false);
}

function renderViralityByPlatform(rows) {
  document.getElementById("virality-bars").innerHTML = renderBarList(rows, "viral_rate", "platform", true, "%");
}

function renderBarList(rows, valueKey, labelKey, normalizeTo100 = false, suffix = "") {
  if (!rows || !rows.length) {
    return `<div class="empty-state">No chart data available.</div>`;
  }
  const max = Math.max(...rows.map((row) => Number(row[valueKey] || 0)), 1);
  return rows.map((row) => {
    const raw = Number(row[valueKey] || 0);
    const width = normalizeTo100 ? Math.min(raw, 100) : (raw / max) * 100;
    const display = suffix ? `${raw.toFixed(1)}${suffix}` : formatInt(raw);
    return `
      <div class="bar-row">
        <div class="bar-head"><span>${row[labelKey]}</span><strong>${display}</strong></div>
        <div class="track"><div class="fill" style="width:${width}%"></div></div>
      </div>
    `;
  }).join("");
}

function renderEvaluation(evaluation, model) {
  renderMatrix(evaluation.confusion_matrix || []);
  renderHistogram(evaluation.probability_histogram || []);
  renderCurveChart("calibration-curve", evaluation.calibration_curve || [], {
    xKey: "mean_predicted_probability",
    yKey: "observed_rate",
    title: "Observed viral rate by prediction bucket",
    diagonal: true
  });
  renderCurveChart("precision-recall-curve", evaluation.precision_recall_curve || [], {
    xKey: "recall",
    yKey: "precision",
    title: "Precision and recall across thresholds",
    diagonal: false,
    markerMode: "spread-duplicates"
  });
  renderTable(document.getElementById("platform-slices-table"), evaluation.slice_metrics?.platform || []);
  renderTable(document.getElementById("niche-slices-table"), evaluation.slice_metrics?.niche || []);
  renderTable(document.getElementById("top-rows-table"), evaluation.top_rows || []);
}

function renderMatrix(matrix) {
  const container = document.getElementById("confusion-matrix");
  if (!matrix.length) {
    container.innerHTML = `<div class="empty-state">No evaluation matrix available.</div>`;
    return;
  }
  const labels = [
    "Actual non-viral -> predicted non-viral",
    "Actual non-viral -> predicted viral",
    "Actual viral -> predicted non-viral",
    "Actual viral -> predicted viral"
  ];
  const values = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
  container.innerHTML = values.map((value, index) => {
    return `<div class="matrix-cell"><strong>${formatInt(value)}</strong><span>${labels[index]}</span></div>`;
  }).join("");
}

function renderFeatureImportance(rows) {
  const container = document.getElementById("feature-importance-list");
  if (!rows || !rows.length) {
    container.innerHTML = `<div class="empty-state">Feature importance will appear after training.</div>`;
    return;
  }
  const max = Math.max(...rows.map((row) => Number(row.importance || 0)), 1);
  container.innerHTML = rows.map((row) => {
    const width = (Number(row.importance || 0) / max) * 100;
    return `
      <div class="bar-row">
        <div class="bar-head"><span>${row.feature}</span><strong>${Number(row.importance).toFixed(3)}</strong></div>
        <div class="track"><div class="fill" style="width:${width}%"></div></div>
      </div>
    `;
  }).join("");
}

function renderHistogram(rows) {
  const container = document.getElementById("probability-histogram");
  if (!rows || !rows.length) {
    container.innerHTML = `<div class="empty-state">No held-out probability distribution available yet.</div>`;
    return;
  }
  const byBucket = {};
  rows.forEach((row) => {
    if (!byBucket[row.bucket]) {
      byBucket[row.bucket] = { bucket: row.bucket, "Non-viral": 0, Viral: 0 };
    }
    byBucket[row.bucket][row.label] = row.count;
  });
  const grouped = Object.values(byBucket);
  const max = Math.max(...grouped.map((row) => row["Non-viral"] + row["Viral"]), 1);
  container.innerHTML = grouped.map((row) => {
    const nonViralHeight = ((row["Non-viral"] || 0) / max) * 100;
    const viralHeight = ((row["Viral"] || 0) / max) * 100;
    return `
      <div class="hist-bar">
        <div class="hist-track">
          <div class="hist-segment non-viral" style="height:${nonViralHeight}%"></div>
          <div class="hist-segment viral" style="height:${viralHeight}%"></div>
        </div>
        <span>${row.bucket}</span>
      </div>
      `;
    }).join("");
  }

function renderCurveChart(containerId, rows, options) {
  const container = document.getElementById(containerId);
  if (!rows || !rows.length) {
    container.innerHTML = `<div class="empty-state">No curve data available for this holdout.</div>`;
    return;
  }

  const ordered = [...rows].sort((left, right) => Number(left[options.xKey] || 0) - Number(right[options.xKey] || 0));
  const width = 520;
  const height = 280;
  const padding = { top: 18, right: 18, bottom: 34, left: 42 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const toX = (value) => padding.left + Number(value || 0) * plotWidth;
  const toY = (value) => height - padding.bottom - Number(value || 0) * plotHeight;

  const diagonal = options.diagonal
    ? `<line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${padding.top}" class="curve-reference" />`
    : "";
  const markers = buildCurveMarkers(ordered, options, toX, toY);
  const polyline = markers
    .map((marker) => `${marker.x.toFixed(1)},${marker.y.toFixed(1)}`)
    .join(" ");
  const circles = markers
    .map((marker) => {
      return `<circle cx="${marker.x.toFixed(1)}" cy="${marker.y.toFixed(1)}" r="3.5" class="curve-point" />`;
    })
    .join("");
  const notes = renderCurveNotes(ordered, options);

  container.innerHTML = `
    <div class="curve-shell">
      <svg viewBox="0 0 ${width} ${height}" class="curve-svg" role="img" aria-label="${options.title}">
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" class="curve-axis" />
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" class="curve-axis" />
        <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" class="curve-grid" />
        <line x1="${padding.left}" y1="${padding.top + plotHeight / 2}" x2="${width - padding.right}" y2="${padding.top + plotHeight / 2}" class="curve-grid" />
        <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" class="curve-grid" />
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" class="curve-grid" />
        <line x1="${padding.left + plotWidth / 2}" y1="${padding.top}" x2="${padding.left + plotWidth / 2}" y2="${height - padding.bottom}" class="curve-grid" />
        <line x1="${width - padding.right}" y1="${padding.top}" x2="${width - padding.right}" y2="${height - padding.bottom}" class="curve-grid" />
        ${diagonal}
        <polyline points="${polyline}" class="curve-line" />
        ${circles}
        <text x="${padding.left}" y="${height - 10}" class="curve-label">0%</text>
        <text x="${padding.left + plotWidth / 2}" y="${height - 10}" class="curve-label">50%</text>
        <text x="${width - padding.right - 6}" y="${height - 10}" text-anchor="end" class="curve-label">100%</text>
        <text x="10" y="${height - padding.bottom}" class="curve-label">0%</text>
        <text x="10" y="${padding.top + plotHeight / 2}" class="curve-label">50%</text>
        <text x="10" y="${padding.top + 6}" class="curve-label">100%</text>
      </svg>
      <div class="curve-meta">
        <div>
          <strong>${options.title}</strong>
          <small>Held-out probability behavior on the latest validation window.</small>
        </div>
        <div class="curve-notes">${notes}</div>
      </div>
    </div>
  `;
}

function buildCurveMarkers(rows, options, toX, toY) {
  const markers = rows.map((point) => ({
    point,
    x: toX(point[options.xKey]),
    y: toY(point[options.yKey])
  }));

  if (options.markerMode !== "spread-duplicates") {
    return markers;
  }

  const groups = new Map();
  markers.forEach((marker) => {
    const key = Number(marker.point[options.xKey] || 0).toFixed(6);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(marker);
  });

  groups.forEach((group) => {
    if (group.length <= 1) {
      return;
    }
    const maxSpread = Math.min(56, Math.max(12, (group.length - 1) * 1.25));
    group.forEach((marker, index) => {
      const ratio = group.length === 1 ? 0 : index / (group.length - 1);
      marker.x -= ratio * maxSpread;
    });
  });

  return markers;
}

function renderCurveNotes(rows, options) {
  return rows
    .slice(-3)
    .map((point) => {
      const rowCount = point.count ? `<span>${formatInt(point.count)} rows</span>` : "";
      return `
        <div class="curve-note">
          <strong>${percent(point[options.yKey])}</strong>
          <small>x ${percent(point[options.xKey])}</small>
          ${rowCount}
        </div>
      `;
    })
    .join("");
}

function renderWatchlist(rows) {
  const list = document.getElementById("watchlist-list");
  list.innerHTML = rows.map((item) => {
    return `<div class="watch-item"><strong>${item.niche}</strong><small>Average five-day engagement rate: ${Number(item.avg_rate).toFixed(3)}</small></div>`;
  }).join("");
}

function renderPreviewTable(rows) {
  renderTable(document.getElementById("preview-table"), rows);
}

function renderTable(table, rows) {
  if (!rows || !rows.length) {
    table.innerHTML = `<tbody><tr><td>No rows available.</td></tr></tbody>`;
    return;
  }
  const columns = Object.keys(rows[0]);
  const thead = `<thead><tr>${columns.map((col) => `<th>${humanize(col)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((row) => `<tr>${columns.map((col) => `<td>${formatCell(row[col])}</td>`).join("")}</tr>`).join("")}</tbody>`;
  table.innerHTML = thead + tbody;
}

function bindPredictionForm() {
  document.getElementById("predict-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = collectPredictionPayload();
    try {
      showFlash("Running prediction...", false);
      const result = await fetchJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      renderPrediction(result);
      showFlash("Prediction updated.", false);
    } catch (error) {
      showFlash(error.message || "Prediction failed.", true);
    }
  });
}

function bindYouTubeLiveForm() {
  document.getElementById("youtube-live-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const videoUrlOrId = document.getElementById("youtube_video_url").value.trim();
    if (!videoUrlOrId) {
      showFlash("Please paste a YouTube URL or video ID first.", true);
      return;
    }

    try {
      showFlash("Loading live YouTube metrics...", false);
      const response = await fetchJSON("/api/live/youtube/video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_url_or_id: videoUrlOrId })
      });
      hydratePredictionForm(response.post);
      renderPrediction(response.prediction);
      showFlash("Live YouTube metrics loaded and scored.", false);
    } catch (error) {
      showFlash(error.message || "Live YouTube lookup failed.", true);
    }
  });
}

function collectPredictionPayload() {
  const date = document.getElementById("post_date").value;
  const time = document.getElementById("post_time").value || "20:00";
  return {
    platform: document.getElementById("platform").value,
    niche: document.getElementById("niche").value,
    caption: document.getElementById("caption").value,
    hashtags: document.getElementById("hashtags").value,
    media_type: document.getElementById("media_type").value,
    post_time: `${date}T${time}:00+00:00`,
    account_follower_count: Number(document.getElementById("account_follower_count").value || 0),
    account_age_days: Number(document.getElementById("account_age_days").value || 0),
    early_window_hours: Number(document.getElementById("early_window_hours").value || 1),
    early_views: Number(document.getElementById("early_views").value || 0),
    early_likes: Number(document.getElementById("early_likes").value || 0),
    early_comments: Number(document.getElementById("early_comments").value || 0),
    early_shares: Number(document.getElementById("early_shares").value || 0)
  };
}

function hydratePredictionForm(post) {
  setSelectValue("platform", post.platform);
  setSelectValue("niche", post.niche);
  setSelectValue("media_type", post.media_type);

  if (post.post_time) {
    const timestamp = new Date(post.post_time);
    if (!Number.isNaN(timestamp.getTime())) {
      document.getElementById("post_date").value = timestamp.toISOString().slice(0, 10);
      document.getElementById("post_time").value = timestamp.toISOString().slice(11, 16);
    }
  }

  document.getElementById("caption").value = post.caption || "";
  document.getElementById("hashtags").value = post.hashtags || "";
  document.getElementById("account_follower_count").value = Number(post.account_follower_count || 0);
  document.getElementById("account_age_days").value = Number(post.account_age_days || 0);
  document.getElementById("early_window_hours").value = snapToStep(post.early_window_hours || 1, 0.25, 0.25);
  document.getElementById("early_views").value = Number(post.early_views || 0);
  document.getElementById("early_likes").value = Number(post.early_likes || 0);
  document.getElementById("early_comments").value = Number(post.early_comments || 0);
  document.getElementById("early_shares").value = Number(post.early_shares || 0);
}

function setSelectValue(id, value) {
  const select = document.getElementById(id);
  if (!select || !value) {
    return;
  }
  const hasOption = Array.from(select.options).some((option) => option.value === value);
  if (!hasOption) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  select.value = value;
}

function renderPrediction(result) {
  const ring = document.getElementById("score-ring");
  ring.style.setProperty("--ring-angle", `${Math.max(0, Math.min(360, result.score * 3.6))}deg`);
  document.getElementById("score-value").textContent = `${result.score}%`;
  document.getElementById("score-bucket").textContent = result.bucket;
  document.getElementById("reasoning-summary").textContent = result.reasoning_summary;
  document.getElementById("signal-list").innerHTML = result.signals.map((signal) => {
    return `
      <div class="signal-row">
        <div class="signal-head"><span>${signal.label}</span><strong>${signal.value.toFixed(1)}</strong></div>
        <div class="track"><div class="fill" style="width:${signal.value}%"></div></div>
      </div>
    `;
  }).join("");
  document.getElementById("recommendation-list").innerHTML = result.recommendations.map((item) => {
    return `<div class="recommendation">${item}</div>`;
  }).join("");
  document.getElementById("positive-factors").innerHTML = renderFactorList(result.positive_factors, false);
  document.getElementById("negative-factors").innerHTML = renderFactorList(result.negative_factors, true);
}

function renderFactorList(rows, negative) {
  if (!rows || !rows.length) {
    return `<div class="empty-state">No factors available yet.</div>`;
  }
  return rows.map((row) => {
    return `
      <article class="factor ${negative ? "negative" : ""}">
        <strong>${row.name}</strong>
        <p>${row.detail}</p>
        <small>Impact ${Number(row.impact).toFixed(2)}</small>
      </article>
    `;
  }).join("");
}

function bindTrainingForms() {
  document.getElementById("demo-train-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      rows: Number(document.getElementById("demo_rows").value || 1800),
      model_name: document.getElementById("demo_model_name").value,
      save_model: document.getElementById("demo_save_model").checked
    };
    try {
      showFlash("Training demo model...", false);
      const response = await fetchJSON("/api/train/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      state.bootstrap = response;
      renderDashboard(response);
      showFlash(response.message || "Demo model trained.", false);
    } catch (error) {
      showFlash(error.message || "Demo training failed.", true);
    }
  });

  document.getElementById("upload-train-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = document.getElementById("train_file").files[0];
    if (!file) {
      showFlash("Please choose a CSV file first.", true);
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", document.getElementById("upload_model_name").value);
    formData.append("save_model", document.getElementById("upload_save_model").checked ? "true" : "false");
    try {
      showFlash("Training from uploaded CSV...", false);
      const response = await fetchJSON("/api/train/upload", {
        method: "POST",
        body: formData
      });
      state.bootstrap = response;
      renderDashboard(response);
      showFlash(response.message || "CSV training complete.", false);
    } catch (error) {
      showFlash(error.message || "CSV training failed.", true);
    }
  });

  document.getElementById("compare-models").addEventListener("click", async () => {
    try {
      showFlash("Benchmarking models...", false);
      const rows = Number(document.getElementById("demo_rows").value || 1800);
      const response = await fetchJSON("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows })
      });
      renderComparison(response.comparison || []);
      showFlash("Model comparison updated.", false);
    } catch (error) {
      showFlash(error.message || "Benchmark failed.", true);
    }
  });
}

function renderComparison(rows) {
  const container = document.getElementById("comparison-chart");
  if (!rows || !rows.length) {
    container.innerHTML = `<div class="empty-state">No comparison data returned.</div>`;
    return;
  }
  const max = Math.max(...rows.map((row) => Number(row.average_precision || 0)), 0.001);
  container.classList.remove("empty-state");
  container.innerHTML = rows.map((row) => {
    const width = (Number(row.average_precision || 0) / max) * 100;
    return `
      <div class="comparison-item">
        <div class="comparison-top">
          <strong>${row.model.replaceAll("_", " ")}</strong>
          <span>${Number(row.average_precision || 0).toFixed(3)} AP</span>
        </div>
        <div class="track"><div class="fill" style="width:${width}%"></div></div>
        <div class="signal-head" style="margin-top:8px;">
          <span>ROC AUC ${Number(row.roc_auc || 0).toFixed(3)}</span>
          <span>F1 ${Number(row.f1 || 0).toFixed(3)}</span>
        </div>
      </div>
    `;
  }).join("");
}

function applyPreset(name) {
  const preset = PRESETS[name];
  document.getElementById("platform").value = preset.platform;
  document.getElementById("niche").value = preset.niche;
  document.getElementById("media_type").value = preset.media_type;
  document.getElementById("caption").value = preset.caption;
  document.getElementById("hashtags").value = preset.hashtags;
  document.getElementById("account_follower_count").value = preset.account_follower_count;
  document.getElementById("account_age_days").value = preset.account_age_days;
  document.getElementById("early_window_hours").value = preset.early_window_hours;
  document.getElementById("early_views").value = preset.early_views;
  document.getElementById("early_likes").value = preset.early_likes;
  document.getElementById("early_comments").value = preset.early_comments;
  document.getElementById("early_shares").value = preset.early_shares;
  document.getElementById("post_time").value = preset.post_time;
  setDefaultDate();
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  let data = null;
  try {
    data = await response.json();
  } catch (error) {
    data = null;
  }
  if (!response.ok) {
    const detail = data?.detail || `Request failed with status ${response.status}`;
    throw new Error(detail);
  }
  return data;
}

function showFlash(message, isError = false, hidden = false) {
  const flash = document.getElementById("flash");
  if (hidden || !message) {
    flash.hidden = true;
    flash.textContent = "";
    flash.classList.remove("error");
    return;
  }
  flash.hidden = false;
  flash.textContent = message;
  flash.classList.toggle("error", isError);
}

function snapToStep(value, step = 1, minimum = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return minimum;
  }
  const safeStep = Number(step) > 0 ? Number(step) : 1;
  const safeMinimum = Number.isFinite(Number(minimum)) ? Number(minimum) : 0;
  const clamped = Math.max(numeric, safeMinimum);
  const units = Math.ceil((clamped - safeMinimum) / safeStep - 1e-9);
  const snapped = safeMinimum + Math.max(units, 0) * safeStep;
  return Number(snapped.toFixed(2));
}

function metricValue(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function formatInt(value) {
  return Number(value || 0).toLocaleString();
}

function formatCell(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? formatInt(value) : value.toFixed(3);
  }
  return String(value);
}

function humanize(value) {
  return String(value).replaceAll("_", " ");
}

function validationLabel(validation) {
  if (!validation || !validation.strategy) {
    return "n/a";
  }
  return humanize(validation.strategy);
}

function validationWindow(validation) {
  if (!validation || !validation.test_start_utc || !validation.test_end_utc) {
    return "n/a";
  }
  const start = validation.test_start_utc.slice(0, 10);
  const end = validation.test_end_utc.slice(0, 10);
  return start === end ? start : `${start} to ${end}`;
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatCell(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? formatInt(value) : value.toFixed(3);
  }
  return String(value);
}
