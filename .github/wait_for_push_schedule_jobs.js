// Poll the current workflow run until every job that On Push / On Schedule
// run on main has completed. Used by the full-CI wait job so "all" stays
// pending until that set is done.
//
// Jobs that only exist on main (codecov secret check, schedule gate) or only
// exist on PRs (all, all_light, this wait job, light-CI notice) are ignored.

const POLL_MS = 15000;
const TIMEOUT_MS = 6 * 60 * 60 * 1000;
const MIN_CI_NESTED_JOBS = 20;
const MAX_REFERENCE_AGE_MS = 14 * 24 * 60 * 60 * 1000;

const IGNORE_EXACT = new Set([
  "all",
  "all_light",
  "Notify Light CI",
  "Check Codecov secret",
  "Check if scheduled CI is needed",
  "Wait for On Push and On Schedule jobs",
  "wait_for_push_schedule",
  // Reusable workflow placeholder when the called workflow is skipped.
  "CI",
]);

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function list_run_jobs(github, owner, repo, run_id, attempt_number) {
  const params = { owner, repo, run_id, per_page: 100 };
  if (attempt_number) {
    try {
      return await github.paginate(
        github.rest.actions.listJobsForWorkflowRunAttempt,
        { ...params, attempt_number }
      );
    } catch (err) {
      // Older octokit builds may not expose the attempt endpoint.
      console.log(
        `listJobsForWorkflowRunAttempt failed (${err.message}); falling back`
      );
    }
  }
  return github.paginate(github.rest.actions.listJobsForWorkflowRun, params);
}

async function reference_jobs(github, core, owner, repo, workflow_id) {
  const { data } = await github.rest.actions.listWorkflowRuns({
    owner,
    repo,
    workflow_id,
    branch: "main",
    per_page: 30,
  });

  for (const run of data.workflow_runs) {
    if (run.status !== "completed") {
      continue;
    }
    const age_ms = Date.now() - Date.parse(run.created_at);
    if (age_ms > MAX_REFERENCE_AGE_MS) {
      continue;
    }
    const jobs = await list_run_jobs(github, owner, repo, run.id);
    const nested_ci = jobs.filter((job) => job.name.startsWith("CI / ")).length;
    if (nested_ci < MIN_CI_NESTED_JOBS) {
      continue;
    }
    core.info(
      `Reference ${workflow_id} run ${run.id} (${run.display_title || run.head_sha}, ${run.conclusion}, ${jobs.length} jobs)`
    );
    return jobs;
  }

  core.warning(`No completed ${workflow_id} run on main with nested CI jobs`);
  return [];
}

function expected_names(jobs, ignore) {
  const names = new Set();
  for (const job of jobs) {
    if (!ignore.has(job.name)) {
      names.add(job.name);
    }
  }
  return names;
}

function format_list(names, limit = 20) {
  const arr = [...names].sort();
  if (arr.length <= limit) {
    return arr.join("\n");
  }
  return `${arr.slice(0, limit).join("\n")}\n... (${arr.length - limit} more)`;
}

module.exports = async function wait_for_push_schedule_jobs({
  github,
  context,
  core,
}) {
  const { owner, repo } = context.repo;
  const started = Date.now();
  const ignore = new Set(IGNORE_EXACT);
  ignore.add(context.job);

  const [push_jobs, schedule_jobs] = await Promise.all([
    reference_jobs(github, core, owner, repo, "on_push_main.yml"),
    reference_jobs(github, core, owner, repo, "on_schedule_main.yml"),
  ]);

  const expected = expected_names([...push_jobs, ...schedule_jobs], ignore);
  if (expected.size === 0) {
    core.setFailed(
      "Could not build the On Push / On Schedule job set from main; refusing to pass the wait job."
    );
    return;
  }

  core.info(
    `Waiting for ${expected.size} On Push / On Schedule jobs in run ${context.runId}`
  );
  core.info(`Expected jobs:\n${format_list(expected, 200)}`);

  while (true) {
    const elapsed_ms = Date.now() - started;

    const jobs = await list_run_jobs(
      github,
      owner,
      repo,
      context.runId,
      context.runAttempt
    );
    const by_name = new Map(jobs.map((job) => [job.name, job]));

    const missing = [];
    const pending = [];
    for (const name of expected) {
      const job = by_name.get(name);
      if (!job) {
        missing.push(name);
      } else if (job.status !== "completed") {
        pending.push(`${name} (${job.status})`);
      }
    }

    const pending_current = jobs
      .filter((job) => !ignore.has(job.name) && job.status !== "completed")
      .map((job) => `${job.name} (${job.status})`);

    core.info(
      `elapsed=${Math.round(elapsed_ms / 1000)}s expected=${expected.size} ` +
        `missing=${missing.length} pending_expected=${pending.length} ` +
        `pending_other=${pending_current.length} current_jobs=${jobs.length}`
    );

    if (elapsed_ms > TIMEOUT_MS) {
      core.setFailed(
        `Timed out after ${Math.round(elapsed_ms / 1000)}s waiting for On Push / On Schedule jobs.\n` +
          `Missing:\n${format_list(missing)}\nPending:\n${format_list(pending)}\n` +
          `Pending other:\n${format_list(pending_current)}`
      );
      return;
    }

    if (missing.length) {
      core.info(`Missing expected jobs:\n${format_list(missing)}`);
    }
    if (pending.length) {
      core.info(`Pending expected jobs:\n${format_list(pending)}`);
    }
    if (pending_current.length) {
      core.info(`Pending other jobs:\n${format_list(pending_current)}`);
    }

    if (
      missing.length === 0 &&
      pending.length === 0 &&
      pending_current.length === 0
    ) {
      core.info("All On Push / On Schedule jobs have completed.");
      return;
    }

    await sleep(POLL_MS);
  }
};
