/* 化学结构简式预览页的前端逻辑。
 *
 * 与 Dashboard 的通信全部走 window.AstrBotPluginPage（bridge SDK）：
 * 不直接拼 /api/... 路径，不加 asset_token，端点是插件内相对路径。
 *
 * 后端三个端点的返回统一是 {"status": "ok", "data": {...}}，
 * 按 bridge 的约定，apiGet/apiPost 会把 data 解出来直接给我们；
 * 这里仍然做一次兼容处理，避免不同版本行为差异。
 */

const bridge = window.AstrBotPluginPage;

const EXAMPLES = [
  "乙醇",
  "葡萄糖",
  "乙酸乙酯",
  "CCO",
  "c1ccccc1",
  "CH3CH(CH3)COOH",
  "CH3(CH2)4CH3",
  "CuSO4·5H2O",
];

const dom = {
  heading: document.getElementById("page-heading"),
  envPill: document.getElementById("env-pill"),
  error: document.getElementById("page-error"),
  form: document.getElementById("render-form"),
  input: document.getElementById("input-field"),
  renderButton: document.getElementById("render-button"),
  examples: document.getElementById("examples"),
  optionImage: document.getElementById("option-image"),
  optionSkeletal: document.getElementById("option-skeletal"),
  optionStructure: document.getElementById("option-structure"),
  downloadButton: document.getElementById("download-button"),
  clearButton: document.getElementById("clear-button"),
  result: document.getElementById("result"),
  resultSource: document.getElementById("result-source"),
  imageHolder: document.getElementById("image-holder"),
  imageCaption: document.getElementById("image-caption"),
  detailList: document.getElementById("detail-list"),
  breakdown: document.getElementById("breakdown"),
  notes: document.getElementById("notes"),
  skeletalBlock: document.getElementById("skeletal-block"),
  skeletalHolder: document.getElementById("skeletal-holder"),
  skeletalMessage: document.getElementById("skeletal-message"),
  structureBlock: document.getElementById("structure-block"),
  structureSummary: document.getElementById("structure-summary"),
  structureHolder: document.getElementById("structure-holder"),
  structureMessage: document.getElementById("structure-message"),
  categorySelect: document.getElementById("category-select"),
  catalogFilter: document.getElementById("catalog-filter"),
  catalogList: document.getElementById("catalog-list"),
  catalogSummary: document.getElementById("catalog-summary"),
};

/** 当前展示的图片地址，供“下载图片”使用。 */
let currentImage = "";

/** 词表数据：类别 -> 条目数组。 */
let catalog = {};

// ---------------------------------------------------------------- 工具函数

/** 兼容处理：bridge 可能已经解包 {status, data}，也可能没有。 */
function unwrap(response) {
  if (
    response &&
    typeof response === "object" &&
    !Array.isArray(response) &&
    "status" in response &&
    "data" in response
  ) {
    if (response.status === "error") {
      throw new Error(response.message || response.error || "请求失败");
    }
    return response.data;
  }
  return response;
}

/** 翻译：优先用插件 i18n，缺失时用兜底文案。 */
function t(key, fallback) {
  try {
    if (bridge && typeof bridge.t === "function") {
      const value = bridge.t(key, fallback);
      if (value) return value;
    }
  } catch (error) {
    /* 忽略：翻译失败不应该影响页面 */
  }
  return fallback;
}

function showError(message) {
  dom.error.textContent = message;
  dom.error.hidden = !message;
}

function clearElement(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function addRow(list, label, value, className) {
  if (value === null || value === undefined || value === "") return;
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (className) dd.className = className;
  if (value instanceof Node) {
    dd.appendChild(value);
  } else {
    dd.textContent = String(value);
  }
  list.appendChild(dt);
  list.appendChild(dd);
}

/** 用 <sub> 渲染下标式子，避免依赖字体的下标字形。 */
function formulaNode(html) {
  const span = document.createElement("span");
  span.className = "formula";
  span.textContent = "";
  if (typeof html === "string" && html) {
    // 后端只返回我们自己生成的 <sub> 标签，这里做一次白名单清洗
    const safe = html.replace(/<(?!\/?sub>)[^>]*>/g, "");
    span.innerHTML = safe;
  }
  return span;
}

function setImage(holder, source, alt) {
  clearElement(holder);
  if (!source) {
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = alt || "无图片";
    holder.appendChild(span);
    return;
  }
  const image = document.createElement("img");
  image.src = source;
  image.alt = alt || "结构简式";
  holder.appendChild(image);
}

// ---------------------------------------------------------------- 渲染结果

function showResult(data) {
  dom.result.hidden = false;
  dom.resultSource.textContent = data.source ? `识别为：${data.source}` : "";

  // 图片
  if (data.image) {
    setImage(dom.imageHolder, data.image, `${data.name || ""} 的结构简式`);
    currentImage = data.image;
    dom.downloadButton.hidden = false;
  } else {
    setImage(dom.imageHolder, "", data.image_error || "该物质没有可绘制的结构简式");
    currentImage = "";
    dom.downloadButton.hidden = true;
  }

  const captionParts = [];
  if (data.formula) captionParts.push(data.formula);
  if (typeof data.molar_mass === "number") {
    captionParts.push(`M = ${data.molar_mass.toFixed(2)}`);
  }
  dom.imageCaption.textContent = captionParts.join("　");

  // 详情
  clearElement(dom.detailList);
  addRow(dom.detailList, "名称", data.name);
  if (data.category) addRow(dom.detailList, "类别", data.category);
  if (data.condensed_html) {
    addRow(dom.detailList, "结构简式", formulaNode(data.condensed_html));
  } else if (data.condensed) {
    addRow(dom.detailList, "结构简式", data.condensed);
  }
  if (data.formula_html) {
    addRow(dom.detailList, "分子式", formulaNode(data.formula_html));
  }
  if (typeof data.molar_mass === "number") {
    addRow(dom.detailList, "相对分子质量", data.molar_mass.toFixed(2));
  }
  if (data.smiles) addRow(dom.detailList, "SMILES", data.smiles);
  if (data.candidates && data.candidates.length) {
    addRow(dom.detailList, "同分异构体", data.candidates.join("、"));
  }

  // 元素质量分数
  clearElement(dom.breakdown);
  (data.breakdown || []).forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "breakdown-item";
    chip.textContent = `${item.symbol} ${item.count} 个 · ${(item.fraction * 100).toFixed(2)}%`;
    dom.breakdown.appendChild(chip);
  });

  // 提示
  clearElement(dom.notes);
  (data.notes || []).forEach((note) => {
    const div = document.createElement("div");
    div.className = "note";
    div.textContent = note;
    dom.notes.appendChild(div);
  });

  // 键线式
  if (data.skeletal) {
    dom.skeletalBlock.hidden = false;
    setImage(dom.skeletalHolder, data.skeletal, "键线式");
    dom.skeletalMessage.textContent = "碳在顶点上不写出来，杂原子标符号，双键画平行线。";
  } else if (data.skeletal_error) {
    dom.skeletalBlock.hidden = false;
    setImage(dom.skeletalHolder, "", "");
    dom.skeletalMessage.textContent = data.skeletal_error;
  } else {
    dom.skeletalBlock.hidden = true;
  }

  // 结构式
  if (data.structure) {
    dom.structureBlock.hidden = false;
    dom.structureHolder.hidden = false;
    dom.structureSummary.textContent = data.structure_engine
      ? `结构式（${data.structure_engine}）`
      : "结构式";
    setImage(dom.structureHolder, data.structure, "结构式");
    dom.structureMessage.textContent = "";
  } else if (data.structure_error) {
    dom.structureBlock.hidden = false;
    dom.structureHolder.hidden = true;
    setImage(dom.structureHolder, "", "");
    dom.structureSummary.textContent = "结构式";
    dom.structureMessage.textContent = data.structure_error;
  } else {
    dom.structureBlock.hidden = true;
  }
}

function showResolveError(error) {
  dom.result.hidden = true;
  const details = (error && error.data && error.data.details) || [];
  const suggestions =
    (error && error.data && error.data.suggestions) || [];
  const parts = [error && error.message ? error.message : "解析失败"];
  if (suggestions.length) parts.push("你是不是想找：" + suggestions.join("、"));
  if (details.length) parts.push(details.map((d) => String(d).split("\n")[0]).join("；"));
  showError(parts.join("\n"));
}

// ---------------------------------------------------------------- 交互

function setBusy(busy) {
  dom.renderButton.disabled = busy;
  dom.renderButton.textContent = busy ? "渲染中…" : "画结构简式";
}

async function render(input) {
  const value = (input || dom.input.value || "").trim();
  if (!value) {
    showError("请先输入物质，例如：乙醇、CCO、CH3CH2OH、C2H6O");
    return;
  }
  showError("");
  setBusy(true);
  try {
    const response = await bridge.apiPost("render", {
      input: value,
      image: dom.optionImage.checked,
      skeletal: dom.optionSkeletal.checked,
      structure: dom.optionStructure.checked,
    });
    showResult(unwrap(response) || {});
  } catch (error) {
    showResolveError(error);
  } finally {
    setBusy(false);
  }
}

function buildExamples() {
  clearElement(dom.examples);
  EXAMPLES.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      dom.input.value = text;
      render(text);
    });
    dom.examples.appendChild(chip);
  });
}

// ---------------------------------------------------------------- 词表

function renderCatalog() {
  const category = dom.categorySelect.value;
  const keyword = (dom.catalogFilter.value || "").trim().toLowerCase();
  const entries = catalog[category] || [];
  const filtered = keyword
    ? entries.filter((entry) => {
        const haystack = [entry.name, ...(entry.aliases || []), entry.smiles || ""]
          .join(" ")
          .toLowerCase();
        return haystack.includes(keyword);
      })
    : entries;

  clearElement(dom.catalogList);
  if (!filtered.length) {
    const empty = document.createElement("span");
    empty.className = "metadata";
    empty.textContent = "没有匹配的物质";
    dom.catalogList.appendChild(empty);
    return;
  }
  filtered.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "catalog-item";
    button.textContent = entry.name;
    button.title = entry.smiles || entry.name;
    button.addEventListener("click", () => {
      dom.input.value = entry.name;
      render(entry.name);
    });
    dom.catalogList.appendChild(button);
  });
}

function fillCatalog(data) {
  catalog = (data && data.catalog) || {};
  const names = Object.keys(catalog);
  clearElement(dom.categorySelect);
  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name}（${catalog[name].length}）`;
    dom.categorySelect.appendChild(option);
  });
  const total = names.reduce((sum, name) => sum + catalog[name].length, 0);
  dom.catalogSummary.textContent = `共 ${total} 条，${names.length} 个类别`;
  renderCatalog();
}

// ---------------------------------------------------------------- 启动

function applyContext(context) {
  if (!context) return;
  if (context.locale) {
    document.documentElement.lang = context.locale;
  }
  const heading = t("pages.structure.heading", "化学结构简式");
  dom.heading.textContent = heading;
  document.title = t("pages.structure.title", "化学结构简式");
}

function describeEnvironment(info) {
  const env = (info && info.environment) || {};
  const parts = [];
  if (env.pillow) parts.push(`Pillow ${env.pillow}`);
  parts.push(env.rdkit_available ? `RDKit ${env.rdkit_version || "可用"}` : "内置键线式引擎");
  dom.envPill.textContent = parts.join(" · ") || "环境未知";
  dom.envPill.classList.toggle("pill-ok", Boolean(env.rdkit_available));

  // 没装 RDKit 也不影响：结构式会退回到内置键线式引擎
  if (!env.rdkit_available) {
    dom.optionStructure.parentElement.title =
      "未安装 RDKit，结构式将由内置键线式引擎绘制（功能不受影响）";
  }
}

async function boot() {
  if (!bridge) {
    showError(
      "没有检测到 AstrBotPluginPage bridge。请确认本页是通过 AstrBot 仪表盘的插件详情页打开，而不是直接双击 HTML 文件。",
    );
    return;
  }

  buildExamples();

  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    render();
  });
  dom.clearButton.addEventListener("click", () => {
    dom.input.value = "";
    dom.result.hidden = true;
    showError("");
    dom.input.focus();
  });
  dom.downloadButton.addEventListener("click", () => {
    if (!currentImage) return;
    const link = document.createElement("a");
    link.href = currentImage;
    link.download = `${(dom.input.value || "structure").replace(/[\\/:*?"<>|]/g, "_")}.png`;
    link.click();
  });
  dom.categorySelect.addEventListener("change", renderCatalog);
  dom.catalogFilter.addEventListener("input", renderCatalog);

  try {
    const context = await bridge.ready();
    applyContext(context);
    if (typeof bridge.onContext === "function") {
      bridge.onContext(applyContext);
    }
  } catch (error) {
    /* 上下文失败不阻塞功能 */
  }

  try {
    describeEnvironment(unwrap(await bridge.apiGet("info")));
  } catch (error) {
    dom.envPill.textContent = "环境信息读取失败";
  }

  try {
    fillCatalog(unwrap(await bridge.apiGet("catalog")));
  } catch (error) {
    dom.catalogSummary.textContent = "词表加载失败";
  }

  dom.input.focus();
}

boot();
