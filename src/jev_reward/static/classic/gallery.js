"use strict";
(() => {
  const scriptURL = document.currentScript.src;
  const mediaURL = new URL("media/", scriptURL);
  const button = document.getElementById("gallery-toggle");
  const restart = document.getElementById("gallery-restart");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let paused = motion.matches;
  let images = [];
  let generation = 0;
  function updatePlayback(reset = false) {
    if (reset) generation++;
    for (const image of images) {
      const url = new URL(paused ? image.dataset.poster : image.dataset.gif, mediaURL);
      if (!paused) url.searchParams.set("play", generation);
      image.src = url.href;
    }
    button.textContent = paused ? "Play animations" : "Pause animations";
    button.setAttribute("aria-pressed", String(paused));
    restart.disabled = paused;
  }
  button.onclick = () => { paused = !paused; updatePlayback(); };
  restart.onclick = () => updatePlayback(true);
  motion.addEventListener("change", event => { paused = event.matches; updatePlayback(); });
  async function load() {
    try {
      const response = await fetch(new URL("gallery.json", mediaURL));
      if (!response.ok) throw new Error("The saved replay gallery could not be loaded.");
      const data = await response.json();
      const table = document.getElementById("checkpoint-gallery");
      const head = table.createTHead().insertRow();
      const heading = document.createElement("th");
      heading.scope = "col"; heading.textContent = "TRAINING"; head.append(heading);
      for (const task of data.tasks) {
        const th = document.createElement("th"); th.scope = "col"; th.textContent = task.name;
        const small = document.createElement("small"); small.textContent = task.env_id;
        th.append(small); head.append(th);
      }
      const labels = {
        untrained: ["0 steps", "Untrained policy"], "10k": ["10,000", "training steps"],
        "30k": ["30,000", "training steps"], final: ["Final policy", "60k / 120k steps"]
      };
      const body = table.createTBody();
      for (const stage of data.stages) {
        const row = body.insertRow(); row.dataset.stage = stage;
        const th = document.createElement("th"); th.scope = "row"; th.textContent = labels[stage][0];
        const sub = document.createElement("small"); sub.textContent = labels[stage][1]; th.append(sub); row.append(th);
        for (const task of data.tasks) {
          const entry = data.entries.find(item => item.task === task.slug && item.stage === stage);
          const cell = row.insertCell();
          const image = document.createElement("img");
          image.width = 320; image.height = 252; image.loading = "lazy"; image.decoding = "async";
          image.dataset.gif = entry.gif; image.dataset.poster = entry.poster;
          image.alt = `${task.name}, ${entry.training_steps.toLocaleString()} training steps, ${entry.success ? "successful" : "unsuccessful"} saved rollout`;
          images.push(image); cell.append(image);
          const caption = document.createElement("p");
          const status = document.createElement("span"); status.className = "outcome" + (entry.success ? " success" : "");
          status.textContent = entry.success ? "Success" : "Not solved";
          caption.append(status, ` · ${entry.replay_steps} moves · total score ${entry.native_return}`);
          cell.append(caption);
        }
      }
      updatePlayback(); button.disabled = false;
    } catch (error) {
      const message = document.getElementById("gallery-error");
      message.textContent = error.message; message.hidden = false;
    }
  }
  load();
})();
