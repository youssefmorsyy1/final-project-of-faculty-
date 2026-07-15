/* ============================================================
   navigation.js  — v2.2.0

   Changes vs v2.1.0
   -----------------
   - Added onNavigate callback property so main.js can hook into
     page transitions and trigger lazy renders.
   - navigateTo() now calls Navigation.onNavigate(pageId) after
     activating the page, enabling the "render on first visit"
     pattern that avoids off-screen canvas sizing bugs.
   ============================================================ */

const PAGE_TITLES = {
  dashboard: "Analytics Dashboard",
  player:    "Player Efficiency & Style Profiling",
  xg:        "Shot Quality — Expected Goals (xG)",
  cohesion:  "Pass Networks (Tactical)",
  injury:    "Injury Risk Prediction",
  winprob:   "Win Probability Modeling",
  eda:       "Exploratory Data Analysis",
  models:    "Models & Methodology",
};

function navigateTo(pageId) {
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));

  const targetPage = document.getElementById("page-" + pageId);
  if (targetPage) {
    targetPage.classList.add("active");
  } else {
    console.warn("[navigation] page not found: page-" + pageId);
  }

  const targetNav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
  if (targetNav) targetNav.classList.add("active");

  const titleEl = document.getElementById("pageTitle");
  if (titleEl) titleEl.textContent = PAGE_TITLES[pageId] || "";

  // Notify main.js so it can lazy-render the newly active page
  if (typeof Navigation.onNavigate === "function") {
    Navigation.onNavigate(pageId);
  }
}

function initNavigation() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      navigateTo(item.dataset.page);
    });
  });
}

const Navigation = {
  init:       initNavigation,
  navigateTo,
  onNavigate: null,   // set by main.js after bootstrap
};