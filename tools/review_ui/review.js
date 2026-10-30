"use strict";
const $ = (id) => document.getElementById(id);
let project = null;
let selectedId = null;
let busy = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function duration(seconds) {
  return typeof seconds === "number" ? `${seconds.toFixed(1).replace(/\.0$/, "")}s` : "Timing to come";
}
function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message;
}
function currentScene() { return project?.scenes.find((scene) => scene.id === selectedId); }

function render() {
  const approved = project.scenes.filter((scene) => scene.reviewStatus === "approved").length;
  $("project-name").textContent = project.name;
  document.title = `${project.name} · Scene review`;
  $("progress").textContent = `${approved} of ${project.scenes.length} approved`;
  $("scene-count").textContent = String(project.scenes.length).padStart(2, "0");
  $("project-meta").textContent = `${project.template} / ${project.phase}`;
  const list = $("scenes");
  list.replaceChildren();
  project.scenes.forEach((scene, index) => {
    const button = element("button", "scene-button");
    button.type = "button";
    button.setAttribute("aria-current", String(scene.id === selectedId));
    button.setAttribute("aria-label", `${index + 1}. ${scene.title}. ${scene.reviewStatus}`);
    const thumb = element("span", "scene-thumb");
    const image = scene.thumbnail || (scene.preview?.kind === "image" ? scene.preview : null);
    if (image) { const img = element("img"); img.src = image.url; img.alt = ""; thumb.append(img); }
    else thumb.textContent = String(index + 1).padStart(2, "0");
    const copy = element("span");
    copy.append(element("span", "scene-name", scene.title));
    const status = element("span", "scene-status", `${duration(scene.durationSeconds)} · ${scene.reviewStatus === "stale" ? "Changed since approval" : scene.reviewStatus}`);
    status.dataset.status = scene.reviewStatus;
    copy.append(status); button.append(thumb, copy);
    button.addEventListener("click", () => { selectedId = scene.id; showError(""); render(); });
    list.append(button);
  });
  $("qc").hidden = project.warnings.length === 0;
  $("qc-count").textContent = String(project.warnings.length);
  $("qc-list").replaceChildren(...project.warnings.map((warning) => {
    const item = element('li', '', `${warning.level}: ${warning.message}`);
    if (typeof warning.timeSeconds === 'number') {
      const seek = element('button', 'quiet', `View at ${warning.timeSeconds.toFixed(1)}s`);
      seek.addEventListener('click', () => {const video = $('preview').querySelector('video'); if (video) video.currentTime = warning.timeSeconds;});
      item.append(seek);
    }
    return item;
  }));
  const scene = currentScene();
  $("approve").disabled = !scene || busy;
  $("save-comment").disabled = !scene || busy;
  if (!scene) {
    $("scene-title").textContent = "Your storyboard is empty.";
    $("preview").replaceChildren(element("p", "", "Add scenes to project.json, then refresh to begin reviewing."));
    return;
  }
  const index = project.scenes.findIndex((item) => item.id === scene.id);
  $("scene-position").textContent = `SCENE ${String(index + 1).padStart(2, "0")} / ${String(project.scenes.length).padStart(2, "0")}`;
  $("scene-title").textContent = scene.title;
  $("scene-meta").textContent = `${scene.type} · ${duration(scene.durationSeconds)} · ${scene.assetStatus}`;
  $("narration").textContent = scene.narration || "No narration planned for this scene.";
  $("preview-name").textContent = scene.preview?.name || "No preview supplied";
  const preview = $("preview"); preview.replaceChildren();
  if (scene.preview?.kind === "video") {
    const video = element("video");
    video.controls = true; video.preload = "metadata"; video.src = scene.preview.url;
    video.setAttribute("aria-label", `${scene.title} preview`);
    if (scene.thumbnail) video.poster = scene.thumbnail.url;
    video.addEventListener("loadedmetadata", () => { video.currentTime = Math.min(scene.previewStart, video.duration || 0); });
    video.addEventListener("timeupdate", () => {
      if (scene.durationSeconds && video.currentTime >= scene.previewStart + scene.durationSeconds) video.pause();
    });
    preview.append(video);
  } else if (scene.preview?.kind === "image") {
    const img = element("img"); img.src = scene.preview.url; img.alt = scene.title; preview.append(img);
  } else {
    const empty = element("p");
    empty.append(element("span", "empty-number", String(index + 1).padStart(2, "0")));
    empty.append(document.createTextNode("No picture yet. Add a scene asset or a final render to review the visuals."));
    preview.append(empty);
  }
  $("issues").replaceChildren(...scene.issues.map((issue) => element("p", "issue", issue)));
  $("audio").replaceChildren();
  $("audio-label").textContent = scene.audio ? "VOICE PREVIEW" : "SCRIPT";
  if (scene.audio) { const audio = element("audio"); audio.src = scene.audio.url; audio.controls = true; audio.preload = "none"; audio.setAttribute("aria-label", `${scene.title} narration`); $("audio").append(audio); }
  const isApproved = scene.reviewStatus === "approved";
  $("approve").textContent = isApproved ? "Scene approved" : "Approve scene";
  $("approve").dataset.approved = String(isApproved);
  $("approve").disabled = isApproved || busy;
  $("reopen").hidden = !isApproved;
  $("review-title").textContent = isApproved ? "This cut is approved." : scene.reviewStatus === "stale" ? "A fresh look needed." : "Ready for a look?";
  $("review-description").textContent = scene.reviewStatus === "stale" ? "The scene, source or assets changed after approval. Review this version before approving again." : "Review the picture and narration, then leave a note or approve this scene.";
  $("notes-count").textContent = String(scene.comments.length);
  $("comments").replaceChildren(...(scene.comments.length ? scene.comments.map((comment) => {
    const article = element("article", "comment"); article.append(element("p", "", comment.text));
    if (typeof comment.timeSeconds === 'number') {
      const seek = element('button', 'quiet', `At ${comment.timeSeconds.toFixed(1)}s`);
      seek.addEventListener('click', () => {const video = $('preview').querySelector('video'); if (video) video.currentTime = comment.timeSeconds;});
      article.append(seek);
    }
    const time = element("time", "", new Date(comment.createdAt).toLocaleString()); time.dateTime = comment.createdAt; article.append(time); return article;
  }) : [element("p", "muted", "No notes yet.")]));
}
async function refresh() {
  try {
    const response = await fetch("/api/project"); const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load project.");
    project = data;
    if (!project.scenes.some((scene) => scene.id === selectedId)) selectedId = project.scenes[0]?.id || null;
    showError(""); render();
  } catch (error) { showError(error.message); }
}
async function mutate(action, text) {
  const scene = currentScene(); if (!scene || busy) return;
  const video = $('preview').querySelector('video');
  const timeSeconds = video ? video.currentTime : null;
  busy = true; render();
  try {
    const response = await fetch("/api/review", {method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": project.token}, body: JSON.stringify({sceneId: scene.id, fingerprint: scene.fingerprint, action, text, timeSeconds})});
    const data = await response.json();
    if (!response.ok) { if (response.status === 409) await refresh(); throw new Error(data.error || "Could not save review."); }
    project = data; showError(""); if (action === "comment") $("comment").value = "";
  } catch (error) { showError(error.message); }
  finally { busy = false; render(); }
}
$("refresh").addEventListener("click", refresh);
$("approve").addEventListener("click", () => mutate("approve"));
$("reopen").addEventListener("click", () => mutate("reopen"));
$("comment-form").addEventListener("submit", (event) => { event.preventDefault(); mutate("comment", $("comment").value); });
refresh();
