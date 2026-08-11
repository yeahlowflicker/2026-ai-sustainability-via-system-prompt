#!/usr/bin/env Rscript

# =============================================================================
# Final preregistered analysis: sustainability system-prompting experiment -> https://osf.io/zsu8v/files/wnhjv
#
# PROJECT LAYOUT EXPECTED:
#   final_analysis.R
#   outputs/exp_metrics.csv
#   analysis/judge_outcomes.csv
#
# RUN H1-H4 ONLY:
#   Rscript final_analysis.R ^
#     --experiment ".\outputs\exp_metrics.csv" ^
#     --output-dir ".\analysis_results"
#
# FINAL RUN INCLUDING H5:
#   Rscript final_analysis.R ^
#     --experiment ".\outputs\exp_metrics.csv" ^
#     --judges ".\analysis\judge_outcomes.csv" ^
#     --require-complete-judges ^
#     --output-dir ".\analysis_results"
#
# Statistical plan:
#   H1-H4:
#       log(outcome) ~ condition * model + (1 | prompt_id)
#       linear mixed-effects model
#
#   H5 QA:
#       qa_correct ~ condition * model + (1 | prompt_id)
#       binomial logistic mixed-effects model
#
#   H5 scored tasks:
#       judge_mean_score ~ condition * model + (1 | prompt_id)
#       linear mixed-effects model
#
# H1-H4 always retain all 600 completed measured model generations.
#
# H5 empty-response policy:
#   - final_content_empty == TRUE and QA task:
#         deterministic incorrect outcome (qa_correct = 0)
#   - final_content_empty == TRUE and 1-5 scored task:
#         deterministic minimum score (judge_mean_score = 1)
#   - judge rows for empty responses are ignored
#   - every NON-EMPTY response must have exactly three valid outcomes,
#     one from each prespecified judge model
#
# Thus model-generation failures/empty outputs are NOT technical failures and
# are NOT rerun. Only missing/malformed judge calls for non-empty responses
# need to be repaired before the final H5 analysis.
# =============================================================================


# -----------------------------------------------------------------------------
# Required packages
# -----------------------------------------------------------------------------

required_packages <- c("lme4", "lmerTest", "emmeans")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required R package(s): ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them in the sustainability conda environment before running."
    ),
    call. = FALSE
  )
}

# Use a fixed, standard denominator-df method for LMM emmeans/joint tests.
emmeans::emm_options(lmer.df = "satterthwaite")

# Sum-to-zero contrasts make omnibus factor tests unambiguous in models
# containing interactions.
options(contrasts = c("contr.sum", "contr.poly"))


# -----------------------------------------------------------------------------
# Constants fixed by the final design
# -----------------------------------------------------------------------------

ALPHA <- 0.05
EXPECTED_PROMPTS <- 100L
EXPECTED_RUNS <- 600L
GENERATION_CAP <- 16384L

EXPECTED_MODELS <- c(
  "granite4.1:8b",
  "gemma4:12b",
  "qwen3.5:9b"
)

EXPECTED_JUDGES <- c(
  "openai/gpt-oss-120b",
  "meta-llama/llama-3.3-70b-instruct",
  "deepseek/deepseek-v4-pro"
)

QA_TASKS <- c(
  "closed_qa",
  "open_qa"
)

SCORED_TASKS <- c(
  "summarization",
  "information_extraction",
  "classification"
)

ALL_TASKS <- c(QA_TASKS, SCORED_TASKS)


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------

parse_args <- function(args) {
  out <- list(
    experiment = NULL,
    judges = NULL,
    output_dir = "analysis_results",
    require_complete_judges = FALSE
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]

    if (key == "--experiment") {
      if (i == length(args)) stop("--experiment requires a path.", call. = FALSE)
      out$experiment <- args[[i + 1L]]
      i <- i + 2L

    } else if (key == "--judges") {
      if (i == length(args)) stop("--judges requires a path.", call. = FALSE)
      out$judges <- args[[i + 1L]]
      i <- i + 2L

    } else if (key == "--output-dir") {
      if (i == length(args)) stop("--output-dir requires a path.", call. = FALSE)
      out$output_dir <- args[[i + 1L]]
      i <- i + 2L

    } else if (key == "--require-complete-judges") {
      out$require_complete_judges <- TRUE
      i <- i + 1L

    } else {
      stop(paste("Unknown argument:", key), call. = FALSE)
    }
  }

  if (is.null(out$experiment)) {
    stop(
      paste0(
        "Missing required --experiment argument.\n",
        "Example:\n",
        'Rscript final_analysis.R --experiment ".\\outputs\\1786315036\\exp_metrics.csv"'
      ),
      call. = FALSE
    )
  }

  out
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

normalize_task <- function(x) {
  x <- trimws(tolower(as.character(x)))
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_+|_+$", "", x)
}


parse_bool <- function(x, field_name = "boolean field") {
  if (is.logical(x)) return(x)

  s <- trimws(tolower(as.character(x)))

  true_values <- c("true", "t", "1", "yes", "y")
  false_values <- c("false", "f", "0", "no", "n")

  result <- rep(NA, length(s))
  result[s %in% true_values] <- TRUE
  result[s %in% false_values] <- FALSE

  if (anyNA(result)) {
    bad <- unique(s[is.na(result)])
    stop(
      paste0(
        "Could not interpret ", field_name, " as boolean. Examples: ",
        paste(head(bad, 5), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  result
}


ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}


write_csv_clean <- function(x, path) {
  ensure_dir(dirname(path))
  write.csv(x, path, row.names = FALSE, na = "")
}


find_ci_columns <- function(df) {
  candidates <- list(
    c("lower.CL", "upper.CL"),
    c("asymp.LCL", "asymp.UCL"),
    c("lower_CL", "upper_CL"),
    c("asymp_LCL", "asymp_UCL")
  )

  for (pair in candidates) {
    if (all(pair %in% names(df))) {
      return(pair)
    }
  }

  stop(
    paste(
      "Could not identify confidence-interval columns.",
      "Available columns:",
      paste(names(df), collapse = ", ")
    ),
    call. = FALSE
  )
}


find_joint_term_column <- function(df) {
  candidates <- c("model term", "model.term", "term", "effect")

  for (candidate in candidates) {
    if (candidate %in% names(df)) return(candidate)
  }

  stop(
    paste(
      "Could not identify joint-test term column.",
      "Available columns:",
      paste(names(df), collapse = ", ")
    ),
    call. = FALSE
  )
}


find_p_column <- function(df) {
  candidates <- c("p.value", "p_value", "pvalue", "p")

  for (candidate in candidates) {
    if (candidate %in% names(df)) return(candidate)
  }

  stop(
    paste(
      "Could not identify p-value column.",
      "Available columns:",
      paste(names(df), collapse = ", ")
    ),
    call. = FALSE
  )
}


extract_joint_p <- function(joint_df, wanted) {
  term_col <- find_joint_term_column(joint_df)
  p_col <- find_p_column(joint_df)

  terms <- tolower(gsub("\\s+", "", as.character(joint_df[[term_col]])))

  if (wanted == "condition") {
    idx <- which(terms == "condition")
  } else if (wanted == "interaction") {
    idx <- which(
      grepl("condition", terms, fixed = TRUE) &
      grepl("model", terms, fixed = TRUE) &
      grepl(":", terms, fixed = TRUE)
    )
  } else {
    stop("Unknown joint-test term requested.", call. = FALSE)
  }

  if (length(idx) != 1L) {
    stop(
      paste0(
        "Could not uniquely identify ", wanted,
        " in omnibus table."
      ),
      call. = FALSE
    )
  }

  as.numeric(joint_df[[p_col]][idx])
}


model_convergence_messages <- function(model) {
  messages <- model@optinfo$conv$lme4$messages
  if (is.null(messages)) character(0) else as.character(messages)
}


save_convergence_log <- function(model, label, output_dir) {
  log_dir <- file.path(output_dir, "logs")
  ensure_dir(log_dir)

  messages <- model_convergence_messages(model)
  singular <- lme4::isSingular(model, tol = 1e-4)

  lines <- c(
    paste("analysis:", label),
    paste("singular_fit:", singular),
    if (length(messages) == 0) {
      "optimizer_messages: none"
    } else {
      c("optimizer_messages:", paste0("  ", messages))
    }
  )

  writeLines(
    lines,
    file.path(log_dir, paste0(label, "_convergence.txt"))
  )

  if (length(messages) > 0) {
    stop(
      paste0(
        label,
        " reported convergence warnings. See its convergence log. ",
        "Do not silently simplify the preregistered model."
      ),
      call. = FALSE
    )
  }
}


# -----------------------------------------------------------------------------
# Experiment data
# -----------------------------------------------------------------------------

prepare_experiment <- function(experiment_csv) {
  if (!file.exists(experiment_csv)) {
    stop(paste("Experiment CSV not found:", experiment_csv), call. = FALSE)
  }

  exp <- read.csv(
    experiment_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  required <- c(
    "run_id",
    "task_category",
    "model_slug",
    "done_reason",
    "output_token_count",
    "total_energy_consumed_kwh",
    "carbon_emissions_grams",
    "latency_s",
    "generation_limit_reached",
    "final_content_empty",
    "generation_failure",
    "reasoning_limit_failure",
    "truncated_response"
  )

  missing <- setdiff(required, names(exp))
  if (length(missing) > 0) {
    stop(
      paste(
        "exp_metrics.csv is missing required columns:",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (nrow(exp) != EXPECTED_RUNS) {
    stop(
      paste0(
        "Expected ", EXPECTED_RUNS,
        " completed measured runs, found ", nrow(exp), "."
      ),
      call. = FALSE
    )
  }

  if (anyDuplicated(exp$run_id)) {
    duplicated_ids <- unique(exp$run_id[duplicated(exp$run_id)])
    stop(
      paste(
        "Duplicate run IDs found:",
        paste(head(duplicated_ids, 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  run_pattern <- "^M([0-9]+)([GS])__UP([0-9]{3})$"
  valid_run_id <- grepl(run_pattern, exp$run_id)

  if (!all(valid_run_id)) {
    stop(
      paste(
        "Invalid run IDs:",
        paste(head(exp$run_id[!valid_run_id], 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  exp$prompt_id <- sub(
    "^M[0-9]+[GS]__(UP[0-9]{3})$",
    "\\1",
    exp$run_id
  )

  exp$model_number_from_run_id <- as.integer(
    sub(
      "^M([0-9]+)[GS]__UP[0-9]{3}$",
      "\\1",
      exp$run_id
    )
  )

  condition_code <- sub(
    "^M[0-9]+([GS])__UP[0-9]{3}$",
    "\\1",
    exp$run_id
  )

  exp$condition <- ifelse(
    condition_code == "G",
    "General",
    "Sustainability"
  )

  exp$task_category <- normalize_task(exp$task_category)

  unexpected_tasks <- setdiff(unique(exp$task_category), ALL_TASKS)
  if (length(unexpected_tasks) > 0) {
    stop(
      paste(
        "Unexpected task categories:",
        paste(unexpected_tasks, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  observed_models <- unique(exp$model_slug)
  if (!setequal(observed_models, EXPECTED_MODELS)) {
    stop(
      paste0(
        "Model slugs do not match the frozen final design.\nExpected: ",
        paste(EXPECTED_MODELS, collapse = ", "),
        "\nObserved: ",
        paste(sort(observed_models), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  model_map <- unique(
    exp[c("model_number_from_run_id", "model_slug")]
  )

  if (
    nrow(model_map) != 3L ||
    length(unique(model_map$model_number_from_run_id)) != 3L ||
    length(unique(model_map$model_slug)) != 3L
  ) {
    stop(
      "Run-ID model numbers do not map one-to-one to the three model slugs.",
      call. = FALSE
    )
  }

  numeric_cols <- c(
    "output_token_count",
    "total_energy_consumed_kwh",
    "carbon_emissions_grams",
    "latency_s"
  )

  for (col in numeric_cols) {
    exp[[col]] <- suppressWarnings(as.numeric(exp[[col]]))

    if (anyNA(exp[[col]])) {
      stop(
        paste(col, "contains missing or non-numeric values."),
        call. = FALSE
      )
    }

    if (any(exp[[col]] <= 0)) {
      stop(
        paste(
          col,
          "contains zero/negative values; log analysis requires positive values."
        ),
        call. = FALSE
      )
    }
  }

  if (any(exp$output_token_count > GENERATION_CAP)) {
    stop(
      paste(
        "output_token_count exceeds the frozen generation cap of",
        GENERATION_CAP
      ),
      call. = FALSE
    )
  }

  flag_cols <- c(
    "generation_limit_reached",
    "final_content_empty",
    "generation_failure",
    "reasoning_limit_failure",
    "truncated_response"
  )

  for (col in flag_cols) {
    exp[[col]] <- parse_bool(exp[[col]], col)
  }

  expected_generation_failure <-
    exp$generation_limit_reached & exp$final_content_empty

  if (!all(exp$generation_failure == expected_generation_failure)) {
    stop(
      paste(
        "generation_failure is inconsistent with",
        "generation_limit_reached AND final_content_empty."
      ),
      call. = FALSE
    )
  }

  if (any(exp$reasoning_limit_failure & !exp$generation_failure)) {
    stop(
      "reasoning_limit_failure must imply generation_failure.",
      call. = FALSE
    )
  }

  expected_truncated <-
    exp$generation_limit_reached & !exp$final_content_empty

  if (!all(exp$truncated_response == expected_truncated)) {
    stop(
      paste(
        "truncated_response is inconsistent with",
        "a length-limited generation that has final content."
      ),
      call. = FALSE
    )
  }

  if (length(unique(exp$prompt_id)) != EXPECTED_PROMPTS) {
    stop(
      paste(
        "Expected",
        EXPECTED_PROMPTS,
        "unique prompts, found",
        length(unique(exp$prompt_id))
      ),
      call. = FALSE
    )
  }

  cell_key <- paste(
    exp$prompt_id,
    exp$model_slug,
    exp$condition,
    sep = "|||"
  )

  if (anyDuplicated(cell_key) || length(unique(cell_key)) != EXPECTED_RUNS) {
    stop(
      "Each prompt x model x condition combination must occur exactly once.",
      call. = FALSE
    )
  }

  prompt_counts <- table(exp$prompt_id)
  if (any(prompt_counts != 6L)) {
    stop(
      "Each prompt must occur exactly six times.",
      call. = FALSE
    )
  }

  prompt_task <- unique(exp[c("prompt_id", "task_category")])

  if (any(table(prompt_task$prompt_id) != 1L)) {
    stop(
      "A prompt is associated with more than one task category.",
      call. = FALSE
    )
  }

  task_prompt_counts <- table(prompt_task$task_category)

  if (
    !setequal(names(task_prompt_counts), ALL_TASKS) ||
    any(task_prompt_counts != 20L)
  ) {
    stop(
      paste0(
        "Prompt-category allocation does not match the final design.\n",
        paste(
          paste(names(task_prompt_counts), task_prompt_counts, sep = "="),
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }

  exp$condition <- factor(
    exp$condition,
    levels = c("General", "Sustainability")
  )

  exp$model <- factor(
    exp$model_slug,
    levels = EXPECTED_MODELS
  )

  exp$prompt_id <- factor(exp$prompt_id)

  exp$token_count <- exp$output_token_count
  exp$energy <- exp$total_energy_consumed_kwh
  exp$carbon <- exp$carbon_emissions_grams
  exp$latency <- exp$latency_s

  exp$log_token_count <- log(exp$token_count)
  exp$log_energy <- log(exp$energy)
  exp$log_carbon <- log(exp$carbon)
  exp$log_latency <- log(exp$latency)

  exp
}


# -----------------------------------------------------------------------------
# Descriptive/QC outputs that do NOT require LLM judges
# -----------------------------------------------------------------------------

write_h1_h4_descriptives <- function(exp, output_dir) {
  outcomes <- list(
    token_count = "Output tokens",
    energy = "Energy consumption (kWh)",
    carbon = "Carbon emissions (g CO2eq)",
    latency = "Latency (s)"
  )

  rows <- list()
  counter <- 1L

  for (outcome in names(outcomes)) {
    for (condition in levels(exp$condition)) {
      for (model in levels(exp$model)) {
        values <- exp[
          exp$condition == condition & exp$model == model,
          outcome
        ]

        rows[[counter]] <- data.frame(
          outcome = outcomes[[outcome]],
          condition = condition,
          model = model,
          n = length(values),
          mean = mean(values),
          sd = sd(values),
          median = median(values),
          iqr = IQR(values),
          min = min(values),
          max = max(values),
          stringsAsFactors = FALSE
        )

        counter <- counter + 1L
      }
    }
  }

  result <- do.call(rbind, rows)

  write_csv_clean(
    result,
    file.path(output_dir, "tables", "Table_1_H1_H4_descriptives.csv")
  )

  result
}


write_generation_qc <- function(exp, output_dir) {
  flags <- c(
    "generation_limit_reached",
    "final_content_empty",
    "generation_failure",
    "reasoning_limit_failure",
    "truncated_response"
  )

  rows <- list()
  counter <- 1L

  for (condition in levels(exp$condition)) {
    for (model in levels(exp$model)) {
      d <- exp[
        exp$condition == condition & exp$model == model,
      ]

      row <- data.frame(
        condition = condition,
        model = model,
        n_runs = nrow(d),
        stringsAsFactors = FALSE
      )

      for (flag in flags) {
        n_flag <- sum(d[[flag]])
        row[[paste0("n_", flag)]] <- n_flag
        row[[paste0("pct_", flag)]] <- 100 * n_flag / nrow(d)
      }

      rows[[counter]] <- row
      counter <- counter + 1L
    }
  }

  qc <- do.call(rbind, rows)

  write_csv_clean(
    qc,
    file.path(output_dir, "tables", "Table_QC_generation_outcomes.csv")
  )

  # IMPORTANT:
  # final_content_empty is included here even if done_reason == "stop".
  # This captures all unusable empty final answers, including the four
  # stop+empty Qwen cases observed in the final experiment.
  flagged <- exp[
    exp$generation_limit_reached |
    exp$final_content_empty |
    exp$generation_failure |
    exp$reasoning_limit_failure |
    exp$truncated_response,
  ]

  flagged_cols <- c(
    "run_id",
    "prompt_id",
    "model_slug",
    "condition",
    "task_category",
    "done_reason",
    "token_count",
    "generation_limit_reached",
    "final_content_empty",
    "generation_failure",
    "reasoning_limit_failure",
    "truncated_response"
  )

  write_csv_clean(
    flagged[flagged_cols],
    file.path(output_dir, "tables", "QC_flagged_generation_runs.csv")
  )

  qc
}


save_generation_qc_figure <- function(qc, output_dir) {
  figure_dir <- file.path(output_dir, "figures")
  ensure_dir(figure_dir)

  values <- matrix(
    0,
    nrow = length(EXPECTED_MODELS),
    ncol = 2,
    dimnames = list(
      EXPECTED_MODELS,
      c("General", "Sustainability")
    )
  )

  for (model in EXPECTED_MODELS) {
    for (condition in c("General", "Sustainability")) {
      row <- qc[
        qc$model == model & qc$condition == condition,
      ]

      if (nrow(row) == 1L) {
        values[model, condition] <- row$pct_generation_limit_reached
      }
    }
  }

  png(
    file.path(figure_dir, "QC_generation_limit_rates.png"),
    width = 1800,
    height = 1200,
    res = 250
  )

  par(mar = c(9, 5, 4, 2) + 0.1)
  barplot(
    t(values),
    beside = TRUE,
    names.arg = EXPECTED_MODELS,
    las = 2,
    ylab = "Runs reaching generation limit (%)",
    main = "Generation-limit outcomes by model and condition",
    legend.text = c("General", "Sustainability"),
    args.legend = list(x = "topright", bty = "n")
  )
  dev.off()

  pdf(
    file.path(figure_dir, "QC_generation_limit_rates.pdf"),
    width = 8,
    height = 5
  )

  par(mar = c(9, 5, 4, 2) + 0.1)
  barplot(
    t(values),
    beside = TRUE,
    names.arg = EXPECTED_MODELS,
    las = 2,
    ylab = "Runs reaching generation limit (%)",
    main = "Generation-limit outcomes by model and condition",
    legend.text = c("General", "Sustainability"),
    args.legend = list(x = "topright", bty = "n")
  )
  dev.off()
}


# -----------------------------------------------------------------------------
# Model plotting
# -----------------------------------------------------------------------------

save_cell_estimate_figure <- function(
  cell_df,
  label,
  output_dir,
  outcome_type
) {
  figure_dir <- file.path(output_dir, "figures")
  ensure_dir(figure_dir)

  ci_cols <- find_ci_columns(cell_df)
  low_col <- ci_cols[[1]]
  high_col <- ci_cols[[2]]

  if (outcome_type == "log_lmm") {
    y <- exp(cell_df$emmean)
    low <- exp(cell_df[[low_col]])
    high <- exp(cell_df[[high_col]])

  } else if (outcome_type == "logistic") {
    if ("prob" %in% names(cell_df)) {
      y <- cell_df$prob
    } else if ("response" %in% names(cell_df)) {
      y <- cell_df$response
    } else {
      stop(
        paste(label, "response-scale emmeans lack a probability column."),
        call. = FALSE
      )
    }

    low <- cell_df[[low_col]]
    high <- cell_df[[high_col]]

  } else {
    y <- cell_df$emmean
    low <- cell_df[[low_col]]
    high <- cell_df[[high_col]]
  }

  plotting <- data.frame(
    condition = as.character(cell_df$condition),
    model = as.character(cell_df$model),
    y = as.numeric(y),
    low = as.numeric(low),
    high = as.numeric(high),
    stringsAsFactors = FALSE
  )

  y_labels <- c(
    H1_output_tokens = "Estimated output tokens",
    H2_energy = "Estimated energy consumption (kWh)",
    H3_carbon = "Estimated carbon emissions (g CO2eq)",
    H4_latency = "Estimated latency (s)",
    H5_QA_correctness = "Estimated probability correct",
    H5_mean_judge_score = "Estimated judge score (1-5)"
  )

  titles <- c(
    H1_output_tokens = "H1: Output tokens",
    H2_energy = "H2: Energy consumption",
    H3_carbon = "H3: Carbon emissions",
    H4_latency = "H4: Latency",
    H5_QA_correctness = "H5: QA correctness",
    H5_mean_judge_score = "H5: Judge score"
  )

  draw_plot <- function() {
    x <- seq_along(EXPECTED_MODELS)
    offsets <- c(General = -0.08, Sustainability = 0.08)
    pch_values <- c(General = 16, Sustainability = 17)

    all_low <- min(plotting$low, na.rm = TRUE)
    all_high <- max(plotting$high, na.rm = TRUE)

    plot(
      NA,
      xlim = c(0.6, length(EXPECTED_MODELS) + 0.4),
      ylim = c(all_low, all_high),
      xaxt = "n",
      xlab = "",
      ylab = y_labels[[label]],
      main = titles[[label]]
    )

    axis(
      1,
      at = x,
      labels = EXPECTED_MODELS,
      las = 2
    )

    for (condition in c("General", "Sustainability")) {
      d <- plotting[
        plotting$condition == condition,
      ]

      d <- d[
        match(EXPECTED_MODELS, d$model),
      ]

      xx <- x + offsets[[condition]]

      points(
        xx,
        d$y,
        pch = pch_values[[condition]]
      )

      arrows(
        xx,
        d$low,
        xx,
        d$high,
        angle = 90,
        code = 3,
        length = 0.05
      )
    }

    legend(
      "topright",
      legend = c("General", "Sustainability"),
      pch = c(16, 17),
      bty = "n"
    )
  }

  png(
    file.path(figure_dir, paste0(label, "_estimated_by_model.png")),
    width = 1800,
    height = 1200,
    res = 250
  )
  par(mar = c(9, 5, 4, 2) + 0.1)
  draw_plot()
  dev.off()

  pdf(
    file.path(figure_dir, paste0(label, "_estimated_by_model.pdf")),
    width = 8,
    height = 5
  )
  par(mar = c(9, 5, 4, 2) + 0.1)
  draw_plot()
  dev.off()
}


save_lmm_diagnostics <- function(model, label, output_dir) {
  diagnostic_dir <- file.path(output_dir, "diagnostics")
  ensure_dir(diagnostic_dir)

  fitted_values <- fitted(model)
  residual_values <- resid(model)

  png(
    file.path(
      diagnostic_dir,
      paste0(label, "_residuals_vs_fitted.png")
    ),
    width = 1500,
    height = 1100,
    res = 250
  )

  plot(
    fitted_values,
    residual_values,
    xlab = "Fitted values",
    ylab = "Residuals",
    main = paste(label, ": residuals vs fitted")
  )
  abline(h = 0, lty = 2)
  dev.off()

  png(
    file.path(
      diagnostic_dir,
      paste0(label, "_qq.png")
    ),
    width = 1500,
    height = 1100,
    res = 250
  )

  qqnorm(
    residual_values,
    main = paste(label, ": residual Q-Q plot")
  )
  qqline(residual_values)
  dev.off()
}


# -----------------------------------------------------------------------------
# H1-H4 linear mixed models
# -----------------------------------------------------------------------------

fit_h1_h4_model <- function(
  exp,
  outcome,
  label,
  output_dir
) {
  formula <- as.formula(
    paste0(
      outcome,
      " ~ condition * model + (1 | prompt_id)"
    )
  )

  model <- lmerTest::lmer(
    formula,
    data = exp,
    REML = TRUE
  )

  save_convergence_log(
    model,
    label,
    output_dir
  )

  coefficient_df <- data.frame(
    term = rownames(summary(model)$coefficients),
    summary(model)$coefficients,
    row.names = NULL,
    check.names = FALSE
  )

  write_csv_clean(
    coefficient_df,
    file.path(
      output_dir,
      "tables",
      paste0(label, "_coefficients.csv")
    )
  )

  joint_df <- as.data.frame(
    emmeans::joint_tests(model)
  )

  write_csv_clean(
    joint_df,
    file.path(
      output_dir,
      "tables",
      paste0(label, "_omnibus.csv")
    )
  )

  p_condition <- extract_joint_p(
    joint_df,
    "condition"
  )

  p_interaction <- extract_joint_p(
    joint_df,
    "interaction"
  )

  cell_emm <- emmeans::emmeans(
    model,
    ~ condition * model
  )

  cell_df <- as.data.frame(
    summary(
      cell_emm,
      infer = c(TRUE, TRUE),
      level = 0.95
    )
  )

  ci_cols <- find_ci_columns(cell_df)

  cell_df$geometric_mean <- exp(cell_df$emmean)
  cell_df$geometric_mean_CI_low <- exp(cell_df[[ci_cols[[1]]]])
  cell_df$geometric_mean_CI_high <- exp(cell_df[[ci_cols[[2]]]])

  write_csv_clean(
    cell_df,
    file.path(
      output_dir,
      "tables",
      paste0(label, "_estimated_marginal_means.csv")
    )
  )

  save_cell_estimate_figure(
    cell_df,
    label,
    output_dir,
    outcome_type = "log_lmm"
  )

  condition_emm <- emmeans::emmeans(
    model,
    ~ condition
  )

  overall_contrast <- emmeans::contrast(
    condition_emm,
    method = list(
      "Sustainability_minus_General" = c(-1, 1)
    ),
    adjust = "none"
  )

  overall_df <- as.data.frame(
    summary(
      overall_contrast,
      infer = c(TRUE, TRUE),
      level = 0.95,
      adjust = "none"
    )
  )

  overall_ci_cols <- find_ci_columns(overall_df)

  overall_df$ratio_geometric_means <- exp(overall_df$estimate)
  overall_df$ratio_CI_low <- exp(overall_df[[overall_ci_cols[[1]]]])
  overall_df$ratio_CI_high <- exp(overall_df[[overall_ci_cols[[2]]]])

  overall_df$percent_change_geometric_mean <-
    (overall_df$ratio_geometric_means - 1) * 100

  overall_df$percent_change_CI_low <-
    (overall_df$ratio_CI_low - 1) * 100

  overall_df$percent_change_CI_high <-
    (overall_df$ratio_CI_high - 1) * 100

  write_csv_clean(
    overall_df,
    file.path(
      output_dir,
      "tables",
      paste0(label, "_overall_condition_contrast.csv")
    )
  )

  if (p_interaction < ALPHA) {
    by_model_emm <- emmeans::emmeans(
      model,
      ~ condition | model
    )

    followup_contrast <- emmeans::contrast(
      by_model_emm,
      method = list(
        "Sustainability_minus_General" = c(-1, 1)
      ),
      adjust = "none"
    )

    followup_df <- as.data.frame(
      summary(
        followup_contrast,
        infer = c(TRUE, TRUE),
        level = 0.95,
        adjust = "none"
      )
    )

    followup_ci_cols <- find_ci_columns(followup_df)

    followup_df$p_value_holm <- p.adjust(
      followup_df$p.value,
      method = "holm"
    )

    followup_df$ratio_geometric_means <- exp(followup_df$estimate)
    followup_df$ratio_CI_low <- exp(
      followup_df[[followup_ci_cols[[1]]]]
    )
    followup_df$ratio_CI_high <- exp(
      followup_df[[followup_ci_cols[[2]]]]
    )
    followup_df$percent_change_geometric_mean <-
      (followup_df$ratio_geometric_means - 1) * 100

    write_csv_clean(
      followup_df,
      file.path(
        output_dir,
        "tables",
        paste0(label, "_followups_by_model_Holm.csv")
      )
    )

  } else {
    writeLines(
      paste0(
        "No model-specific follow-ups were run because the ",
        "condition:model omnibus interaction p-value was ",
        format(p_interaction, digits = 6),
        ", not below alpha=",
        ALPHA,
        "."
      ),
      file.path(
        output_dir,
        "tables",
        paste0(label, "_no_followups.txt")
      )
    )
  }

  save_lmm_diagnostics(
    model,
    label,
    output_dir
  )

  data.frame(
    analysis = label,
    outcome = outcome,
    n = nrow(exp),
    effect_scale = "ratio of geometric means",
    effect = overall_df$ratio_geometric_means[[1]],
    ci_low = overall_df$ratio_CI_low[[1]],
    ci_high = overall_df$ratio_CI_high[[1]],
    percent_change = overall_df$percent_change_geometric_mean[[1]],
    percent_ci_low = overall_df$percent_change_CI_low[[1]],
    percent_ci_high = overall_df$percent_change_CI_high[[1]],
    condition_omnibus_p = p_condition,
    interaction_omnibus_p = p_interaction,
    overall_contrast_p = overall_df$p.value[[1]],
    stringsAsFactors = FALSE
  )
}


# -----------------------------------------------------------------------------
# Judge data
#
# Empty final responses do not require LLM judging:
#   QA -> incorrect (0)
#   scored tasks -> minimum score (1)
#
# Every non-empty response requires exactly one valid result from each of the
# three prespecified judge models.
# -----------------------------------------------------------------------------

prepare_judges <- function(judge_csv, exp, output_dir) {
  if (!file.exists(judge_csv)) {
    stop(
      paste("Judge CSV not found:", judge_csv),
      call. = FALSE
    )
  }

  j <- read.csv(
    judge_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  required <- c(
    "Run ID",
    "LLM Judge Model",
    "Task",
    "Outcome"
  )

  missing <- setdiff(required, names(j))
  if (length(missing) > 0) {
    stop(
      paste(
        "Judge CSV is missing required columns:",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # Remove fully blank rows, if any.
  j <- j[
    !apply(
      j,
      1,
      function(row) all(is.na(row) | trimws(as.character(row)) == "")
    ),
  ]

  j$run_id <- trimws(as.character(j[["Run ID"]]))
  j$judge_model <- trimws(as.character(j[["LLM Judge Model"]]))
  j$judge_task_category <- normalize_task(j[["Task"]])
  j$outcome_raw <- trimws(as.character(j[["Outcome"]]))

  # Reject judge rows that do not correspond to a measured experiment run.
  unknown_run_ids <- setdiff(
    unique(j$run_id),
    as.character(exp$run_id)
  )

  if (length(unknown_run_ids) > 0) {
    stop(
      paste(
        "Judge CSV contains Run IDs that are absent from exp_metrics.csv:",
        paste(head(unknown_run_ids, 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  unexpected_judges <- setdiff(
    unique(j$judge_model),
    EXPECTED_JUDGES
  )

  if (length(unexpected_judges) > 0) {
    stop(
      paste(
        "Unexpected judge model(s):",
        paste(unexpected_judges, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  unexpected_tasks <- setdiff(
    unique(j$judge_task_category),
    ALL_TASKS
  )

  if (length(unexpected_tasks) > 0) {
    stop(
      paste(
        "Unexpected judge task categories:",
        paste(unexpected_tasks, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Separate empty vs non-empty model responses.
  # Judge rows for empty responses are deliberately ignored.
  # ---------------------------------------------------------------------------

  exp_run_ids <- as.character(exp$run_id)

  empty_ids <- exp_run_ids[exp$final_content_empty]
  nonempty_ids <- exp_run_ids[!exp$final_content_empty]

  j_empty <- j[
    j$run_id %in% empty_ids,
  ]

  if (nrow(j_empty) > 0) {
    write_csv_clean(
      j_empty[
        c(
          "run_id",
          "judge_model",
          "judge_task_category",
          "outcome_raw"
        )
      ],
      file.path(
        output_dir,
        "tables",
        "H5_ignored_judgments_for_empty_responses.csv"
      )
    )
  }

  j_nonempty <- j[
    j$run_id %in% nonempty_ids,
  ]

  # One outcome at most for each required Run ID x judge pair.
  pair_key <- paste(
    j_nonempty$run_id,
    j_nonempty$judge_model,
    sep = "|||"
  )

  if (anyDuplicated(pair_key)) {
    duplicated_pairs <- unique(
      pair_key[duplicated(pair_key)]
    )

    stop(
      paste(
        "Duplicate non-empty (Run ID, LLM Judge Model) pairs found:",
        paste(head(duplicated_pairs, 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # Judge task labels must agree with exp_metrics.csv for every present row.
  task_lookup <- setNames(
    as.character(exp$task_category),
    exp_run_ids
  )

  expected_task_for_judge_row <- task_lookup[j_nonempty$run_id]

  task_mismatch <-
    j_nonempty$judge_task_category != expected_task_for_judge_row

  if (any(task_mismatch)) {
    bad_rows <- unique(
      j_nonempty$run_id[task_mismatch]
    )

    stop(
      paste(
        "Experiment and judge task categories disagree for:",
        paste(head(bad_rows, 10), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Require the complete prespecified 3-judge panel ONLY for non-empty runs.
  # ---------------------------------------------------------------------------

  expected_grid <- expand.grid(
    run_id = nonempty_ids,
    judge_model = EXPECTED_JUDGES,
    stringsAsFactors = FALSE
  )

  expected_key <- paste(
    expected_grid$run_id,
    expected_grid$judge_model,
    sep = "|||"
  )

  observed_key <- paste(
    j_nonempty$run_id,
    j_nonempty$judge_model,
    sep = "|||"
  )

  missing_key <- setdiff(
    expected_key,
    observed_key
  )

  if (length(missing_key) > 0) {
    missing_grid <- expected_grid[
      expected_key %in% missing_key,
    ]

    exp_task_lookup <- setNames(
      as.character(exp$task_category),
      exp_run_ids
    )

    missing_grid$task_category <-
      exp_task_lookup[missing_grid$run_id]

    write_csv_clean(
      missing_grid,
      file.path(
        output_dir,
        "tables",
        "H5_missing_required_judgments.csv"
      )
    )

    stop(
      paste0(
        "H5 requires three valid judges for each NON-EMPTY response. ",
        "Missing ", nrow(missing_grid),
        " required judge outcome(s). See tables/H5_missing_required_judgments.csv. ",
        "Do NOT rerun model inference; repair only those judge call(s)."
      ),
      call. = FALSE
    )
  }

  expected_nonempty_rows <-
    length(nonempty_ids) * length(EXPECTED_JUDGES)

  if (nrow(j_nonempty) != expected_nonempty_rows) {
    stop(
      paste0(
        "Unexpected non-empty judge row count. Expected ",
        expected_nonempty_rows,
        ", found ",
        nrow(j_nonempty),
        "."
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Validate judge output format on non-empty responses.
  # ---------------------------------------------------------------------------

  qa_mask <-
    j_nonempty$judge_task_category %in% QA_TASKS

  scored_mask <-
    j_nonempty$judge_task_category %in% SCORED_TASKS

  qa_values <- toupper(
    j_nonempty$outcome_raw[qa_mask]
  )

  if (any(!qa_values %in% c("TRUE", "FALSE"))) {
    bad <- unique(
      j_nonempty$outcome_raw[qa_mask][
        !qa_values %in% c("TRUE", "FALSE")
      ]
    )

    stop(
      paste(
        "QA judge outcomes for non-empty responses must be exactly TRUE/FALSE.",
        "Invalid examples:",
        paste(head(bad, 5), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  score_values <- suppressWarnings(
    as.numeric(
      j_nonempty$outcome_raw[scored_mask]
    )
  )

  if (
    anyNA(score_values) ||
    any(score_values < 1 | score_values > 5) ||
    any(score_values %% 1 != 0)
  ) {
    stop(
      paste(
        "Scored-task judge outcomes for non-empty responses",
        "must be integer values from 1 to 5."
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Aggregate the complete 3-judge panel for non-empty responses.
  # ---------------------------------------------------------------------------

  panel_rows <- vector(
    "list",
    length(nonempty_ids)
  )

  for (i in seq_along(nonempty_ids)) {
    run_id <- nonempty_ids[[i]]

    g <- j_nonempty[
      j_nonempty$run_id == run_id,
    ]

    if (
      nrow(g) != 3L ||
      length(unique(g$judge_model)) != 3L ||
      !setequal(g$judge_model, EXPECTED_JUDGES)
    ) {
      stop(
        paste(
          "Non-empty response does not have the complete prespecified panel:",
          run_id
        ),
        call. = FALSE
      )
    }

    task <- unique(g$judge_task_category)

    if (length(task) != 1L) {
      stop(
        paste(
          "Judges disagree on task category for",
          run_id
        ),
        call. = FALSE
      )
    }

    task <- task[[1]]

    if (task %in% QA_TASKS) {
      votes <- toupper(g$outcome_raw) == "TRUE"

      panel_rows[[i]] <- data.frame(
        run_id = run_id,
        judge_task_category = task,
        qa_correct = as.integer(sum(votes) >= 2L),
        judge_mean_score = NA_real_,
        n_judges = 3L,
        h5_scoring_source = "three_judge_panel",
        stringsAsFactors = FALSE
      )

    } else {
      scores <- as.numeric(g$outcome_raw)

      panel_rows[[i]] <- data.frame(
        run_id = run_id,
        judge_task_category = task,
        qa_correct = NA_integer_,
        judge_mean_score = mean(scores),
        n_judges = 3L,
        h5_scoring_source = "three_judge_panel",
        stringsAsFactors = FALSE
      )
    }
  }

  panel_outcomes <- do.call(
    rbind,
    panel_rows
  )

  # ---------------------------------------------------------------------------
  # Deterministically score empty final responses.
  # ---------------------------------------------------------------------------

  empty_exp <- exp[
    exp$final_content_empty,
  ]

  empty_rows <- vector(
    "list",
    nrow(empty_exp)
  )

  if (nrow(empty_exp) > 0) {
    for (i in seq_len(nrow(empty_exp))) {
      run_id <- as.character(
        empty_exp$run_id[[i]]
      )

      task <- as.character(
        empty_exp$task_category[[i]]
      )

      if (task %in% QA_TASKS) {
        empty_rows[[i]] <- data.frame(
          run_id = run_id,
          judge_task_category = task,
          qa_correct = 0L,
          judge_mean_score = NA_real_,
          n_judges = 0L,
          h5_scoring_source = "deterministic_empty_response",
          stringsAsFactors = FALSE
        )

      } else {
        empty_rows[[i]] <- data.frame(
          run_id = run_id,
          judge_task_category = task,
          qa_correct = NA_integer_,
          judge_mean_score = 1,
          n_judges = 0L,
          h5_scoring_source = "deterministic_empty_response",
          stringsAsFactors = FALSE
        )
      }
    }

    empty_outcomes <- do.call(
      rbind,
      empty_rows
    )

    h5_outcomes <- rbind(
      panel_outcomes,
      empty_outcomes
    )

  } else {
    h5_outcomes <- panel_outcomes
  }

  if (
    nrow(h5_outcomes) != EXPECTED_RUNS ||
    anyDuplicated(h5_outcomes$run_id)
  ) {
    stop(
      paste(
        "Internal H5 aggregation error:",
        "expected exactly one H5 outcome for each of the 600 runs."
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Merge H5 outcomes with the full 600-run experiment data.
  # ---------------------------------------------------------------------------

  exp$run_id <- as.character(
    exp$run_id
  )

  merged <- merge(
    exp,
    h5_outcomes,
    by = "run_id",
    all.x = TRUE,
    all.y = FALSE,
    sort = FALSE
  )

  if (
    nrow(merged) != EXPECTED_RUNS ||
    anyNA(merged$h5_scoring_source)
  ) {
    stop(
      "Every experiment run must have exactly one final H5 outcome.",
      call. = FALSE
    )
  }

  mismatch <-
    as.character(merged$task_category) !=
    merged$judge_task_category

  if (any(mismatch)) {
    stop(
      paste(
        "Experiment and final H5 task categories disagree for:",
        paste(
          head(merged$run_id[mismatch], 10),
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }

  # Empty-response rule should be exactly represented after merge.
  empty_qa <-
    merged$final_content_empty &
    merged$task_category %in% QA_TASKS

  if (
    any(
      empty_qa &
      (
        merged$qa_correct != 0L |
        merged$h5_scoring_source != "deterministic_empty_response"
      )
    )
  ) {
    stop(
      "Internal error applying deterministic empty-response rule for QA.",
      call. = FALSE
    )
  }

  empty_scored <-
    merged$final_content_empty &
    merged$task_category %in% SCORED_TASKS

  if (
    any(
      empty_scored &
      (
        merged$judge_mean_score != 1 |
        merged$h5_scoring_source != "deterministic_empty_response"
      )
    )
  ) {
    stop(
      "Internal error applying deterministic empty-response rule for scored tasks.",
      call. = FALSE
    )
  }

  # Restore factor levels after merge.
  merged$condition <- factor(
    merged$condition,
    levels = c(
      "General",
      "Sustainability"
    )
  )

  merged$model <- factor(
    merged$model_slug,
    levels = EXPECTED_MODELS
  )

  merged$prompt_id <- factor(
    merged$prompt_id
  )

  # Audit table showing where H5 outcomes came from.
  source_qc <- aggregate(
    x = list(
      n_responses = rep(
        1L,
        nrow(merged)
      )
    ),
    by = list(
      h5_scoring_source = merged$h5_scoring_source,
      task_category = as.character(merged$task_category),
      condition = as.character(merged$condition),
      model = as.character(merged$model)
    ),
    FUN = sum
  )

  write_csv_clean(
    source_qc,
    file.path(
      output_dir,
      "tables",
      "Table_QC_H5_scoring_sources.csv"
    )
  )

  # A simple run-level audit is useful for the paper/reproducibility archive.
  write_csv_clean(
    merged[
      c(
        "run_id",
        "task_category",
        "model_slug",
        "condition",
        "final_content_empty",
        "h5_scoring_source",
        "n_judges",
        "qa_correct",
        "judge_mean_score"
      )
    ],
    file.path(
      output_dir,
      "tables",
      "H5_run_level_outcomes.csv"
    )
  )

  merged
}


write_h5_descriptives <- function(data, output_dir) {
  qa <- data[
    data$task_category %in% QA_TASKS,
  ]

  qa_rows <- list()
  counter <- 1L

  for (condition in levels(data$condition)) {
    for (model in levels(data$model)) {
      d <- qa[
        qa$condition == condition & qa$model == model,
      ]

      qa_rows[[counter]] <- data.frame(
        condition = condition,
        model = model,
        n = nrow(d),
        n_correct = sum(d$qa_correct),
        accuracy = mean(d$qa_correct),
        accuracy_percent = 100 * mean(d$qa_correct),
        stringsAsFactors = FALSE
      )

      counter <- counter + 1L
    }
  }

  write_csv_clean(
    do.call(rbind, qa_rows),
    file.path(
      output_dir,
      "tables",
      "Table_2_H5_QA_descriptives.csv"
    )
  )

  scored <- data[
    data$task_category %in% SCORED_TASKS,
  ]

  score_rows <- list()
  counter <- 1L

  for (condition in levels(data$condition)) {
    for (model in levels(data$model)) {
      d <- scored[
        scored$condition == condition & scored$model == model,
      ]

      values <- d$judge_mean_score

      score_rows[[counter]] <- data.frame(
        condition = condition,
        model = model,
        n = length(values),
        mean = mean(values),
        sd = sd(values),
        median = median(values),
        stringsAsFactors = FALSE
      )

      counter <- counter + 1L
    }
  }

  write_csv_clean(
    do.call(rbind, score_rows),
    file.path(
      output_dir,
      "tables",
      "Table_3_H5_score_descriptives.csv"
    )
  )
}


# -----------------------------------------------------------------------------
# H5 models
# -----------------------------------------------------------------------------

fit_h5_qa <- function(data, output_dir) {
  qa <- data[
    data$task_category %in% QA_TASKS,
  ]

  model <- lme4::glmer(
    qa_correct ~ condition * model + (1 | prompt_id),
    data = qa,
    family = binomial(link = "logit")
  )

  save_convergence_log(
    model,
    "H5_QA_correctness",
    output_dir
  )

  coefficient_df <- data.frame(
    term = rownames(summary(model)$coefficients),
    summary(model)$coefficients,
    row.names = NULL,
    check.names = FALSE
  )

  write_csv_clean(
    coefficient_df,
    file.path(
      output_dir,
      "tables",
      "H5_QA_correctness_coefficients.csv"
    )
  )

  joint_df <- as.data.frame(
    emmeans::joint_tests(model)
  )

  write_csv_clean(
    joint_df,
    file.path(
      output_dir,
      "tables",
      "H5_QA_correctness_omnibus.csv"
    )
  )

  p_condition <- extract_joint_p(
    joint_df,
    "condition"
  )

  p_interaction <- extract_joint_p(
    joint_df,
    "interaction"
  )

  cell_response <- emmeans::emmeans(
    model,
    ~ condition * model,
    type = "response"
  )

  cell_df <- as.data.frame(
    summary(
      cell_response,
      infer = c(TRUE, TRUE),
      level = 0.95,
      type = "response"
    )
  )

  write_csv_clean(
    cell_df,
    file.path(
      output_dir,
      "tables",
      "H5_QA_correctness_estimated_marginal_means.csv"
    )
  )

  save_cell_estimate_figure(
    cell_df,
    "H5_QA_correctness",
    output_dir,
    outcome_type = "logistic"
  )

  condition_link <- emmeans::emmeans(
    model,
    ~ condition,
    type = "link"
  )

  overall <- emmeans::contrast(
    condition_link,
    method = list(
      "Sustainability_minus_General" = c(-1, 1)
    ),
    adjust = "none"
  )

  overall_df <- as.data.frame(
    summary(
      overall,
      infer = c(TRUE, TRUE),
      level = 0.95,
      adjust = "none"
    )
  )

  ci_cols <- find_ci_columns(overall_df)

  overall_df$odds_ratio <- exp(overall_df$estimate)
  overall_df$odds_ratio_CI_low <- exp(overall_df[[ci_cols[[1]]]])
  overall_df$odds_ratio_CI_high <- exp(overall_df[[ci_cols[[2]]]])

  write_csv_clean(
    overall_df,
    file.path(
      output_dir,
      "tables",
      "H5_QA_correctness_overall_condition_contrast.csv"
    )
  )

  condition_probabilities <- as.data.frame(
    summary(
      emmeans::emmeans(
        model,
        ~ condition,
        type = "response"
      ),
      infer = c(TRUE, TRUE),
      level = 0.95,
      type = "response"
    )
  )

  write_csv_clean(
    condition_probabilities,
    file.path(
      output_dir,
      "tables",
      "H5_QA_correctness_condition_probabilities.csv"
    )
  )

  if (p_interaction < ALPHA) {
    by_model <- emmeans::emmeans(
      model,
      ~ condition | model,
      type = "link"
    )

    followup <- emmeans::contrast(
      by_model,
      method = list(
        "Sustainability_minus_General" = c(-1, 1)
      ),
      adjust = "none"
    )

    followup_df <- as.data.frame(
      summary(
        followup,
        infer = c(TRUE, TRUE),
        level = 0.95,
        adjust = "none"
      )
    )

    followup_ci <- find_ci_columns(followup_df)

    followup_df$p_value_holm <- p.adjust(
      followup_df$p.value,
      method = "holm"
    )

    followup_df$odds_ratio <- exp(followup_df$estimate)
    followup_df$odds_ratio_CI_low <- exp(
      followup_df[[followup_ci[[1]]]]
    )
    followup_df$odds_ratio_CI_high <- exp(
      followup_df[[followup_ci[[2]]]]
    )

    write_csv_clean(
      followup_df,
      file.path(
        output_dir,
        "tables",
        "H5_QA_correctness_followups_by_model_Holm.csv"
      )
    )

  } else {
    writeLines(
      paste0(
        "No model-specific follow-ups were run because the ",
        "condition:model omnibus interaction p-value was ",
        format(p_interaction, digits = 6),
        ", not below alpha=",
        ALPHA,
        "."
      ),
      file.path(
        output_dir,
        "tables",
        "H5_QA_correctness_no_followups.txt"
      )
    )
  }

  data.frame(
    analysis = "H5_QA_correctness",
    outcome = "qa_correct",
    n = nrow(qa),
    effect_scale = "odds ratio",
    effect = overall_df$odds_ratio[[1]],
    ci_low = overall_df$odds_ratio_CI_low[[1]],
    ci_high = overall_df$odds_ratio_CI_high[[1]],
    percent_change = NA_real_,
    percent_ci_low = NA_real_,
    percent_ci_high = NA_real_,
    condition_omnibus_p = p_condition,
    interaction_omnibus_p = p_interaction,
    overall_contrast_p = overall_df$p.value[[1]],
    stringsAsFactors = FALSE
  )
}


fit_h5_score <- function(data, output_dir) {
  scored <- data[
    data$task_category %in% SCORED_TASKS,
  ]

  model <- lmerTest::lmer(
    judge_mean_score ~ condition * model + (1 | prompt_id),
    data = scored,
    REML = TRUE
  )

  save_convergence_log(
    model,
    "H5_mean_judge_score",
    output_dir
  )

  coefficient_df <- data.frame(
    term = rownames(summary(model)$coefficients),
    summary(model)$coefficients,
    row.names = NULL,
    check.names = FALSE
  )

  write_csv_clean(
    coefficient_df,
    file.path(
      output_dir,
      "tables",
      "H5_mean_judge_score_coefficients.csv"
    )
  )

  joint_df <- as.data.frame(
    emmeans::joint_tests(model)
  )

  write_csv_clean(
    joint_df,
    file.path(
      output_dir,
      "tables",
      "H5_mean_judge_score_omnibus.csv"
    )
  )

  p_condition <- extract_joint_p(
    joint_df,
    "condition"
  )

  p_interaction <- extract_joint_p(
    joint_df,
    "interaction"
  )

  cell <- emmeans::emmeans(
    model,
    ~ condition * model
  )

  cell_df <- as.data.frame(
    summary(
      cell,
      infer = c(TRUE, TRUE),
      level = 0.95
    )
  )

  write_csv_clean(
    cell_df,
    file.path(
      output_dir,
      "tables",
      "H5_mean_judge_score_estimated_marginal_means.csv"
    )
  )

  save_cell_estimate_figure(
    cell_df,
    "H5_mean_judge_score",
    output_dir,
    outcome_type = "lmm"
  )

  condition_emm <- emmeans::emmeans(
    model,
    ~ condition
  )

  overall <- emmeans::contrast(
    condition_emm,
    method = list(
      "Sustainability_minus_General" = c(-1, 1)
    ),
    adjust = "none"
  )

  overall_df <- as.data.frame(
    summary(
      overall,
      infer = c(TRUE, TRUE),
      level = 0.95,
      adjust = "none"
    )
  )

  ci_cols <- find_ci_columns(overall_df)

  write_csv_clean(
    overall_df,
    file.path(
      output_dir,
      "tables",
      "H5_mean_judge_score_overall_condition_contrast.csv"
    )
  )

  if (p_interaction < ALPHA) {
    by_model <- emmeans::emmeans(
      model,
      ~ condition | model
    )

    followup <- emmeans::contrast(
      by_model,
      method = list(
        "Sustainability_minus_General" = c(-1, 1)
      ),
      adjust = "none"
    )

    followup_df <- as.data.frame(
      summary(
        followup,
        infer = c(TRUE, TRUE),
        level = 0.95,
        adjust = "none"
      )
    )

    followup_df$p_value_holm <- p.adjust(
      followup_df$p.value,
      method = "holm"
    )

    write_csv_clean(
      followup_df,
      file.path(
        output_dir,
        "tables",
        "H5_mean_judge_score_followups_by_model_Holm.csv"
      )
    )

  } else {
    writeLines(
      paste0(
        "No model-specific follow-ups were run because the ",
        "condition:model omnibus interaction p-value was ",
        format(p_interaction, digits = 6),
        ", not below alpha=",
        ALPHA,
        "."
      ),
      file.path(
        output_dir,
        "tables",
        "H5_mean_judge_score_no_followups.txt"
      )
    )
  }

  save_lmm_diagnostics(
    model,
    "H5_mean_judge_score",
    output_dir
  )

  data.frame(
    analysis = "H5_mean_judge_score",
    outcome = "judge_mean_score",
    n = nrow(scored),
    effect_scale = "Sustainability - General mean-score difference",
    effect = overall_df$estimate[[1]],
    ci_low = overall_df[[ci_cols[[1]]]][[1]],
    ci_high = overall_df[[ci_cols[[2]]]][[1]],
    percent_change = NA_real_,
    percent_ci_low = NA_real_,
    percent_ci_high = NA_real_,
    condition_omnibus_p = p_condition,
    interaction_omnibus_p = p_interaction,
    overall_contrast_p = overall_df$p.value[[1]],
    stringsAsFactors = FALSE
  )
}


# -----------------------------------------------------------------------------
# Final paper-oriented summary outputs
# -----------------------------------------------------------------------------

add_interpretation <- function(summary_df) {
  summary_df$preregistered_interpretation <- ""

  h1_h4 <- grepl(
    "^H[1-4]",
    summary_df$analysis
  )

  summary_df$preregistered_interpretation[h1_h4] <- ifelse(
    summary_df$condition_omnibus_p[h1_h4] < ALPHA &
    summary_df$effect[h1_h4] < 1,
    "Significant effect in predicted lower direction",
    "Predicted reduction not supported at alpha=.05"
  )

  h5 <- grepl(
    "^H5",
    summary_df$analysis
  )

  if (any(h5)) {
    summary_df$preregistered_interpretation[h5] <- ifelse(
      summary_df$condition_omnibus_p[h5] < ALPHA,
      "Statistically detectable condition difference; inspect direction",
      paste(
        "No statistically detectable condition difference.",
        "This is NOT proof of equivalence."
      )
    )
  }

  summary_df
}


write_paper_primary_table <- function(summary_df, output_dir) {
  outcome_names <- c(
    H1_output_tokens = "Output token count",
    H2_energy = "Energy consumption",
    H3_carbon = "Carbon emissions",
    H4_latency = "Latency",
    H5_QA_correctness = "QA correctness",
    H5_mean_judge_score = "Judge mean score"
  )

  paper <- summary_df
  paper$paper_outcome <- outcome_names[paper$analysis]

  keep <- c(
    "analysis",
    "paper_outcome",
    "n",
    "effect_scale",
    "effect",
    "ci_low",
    "ci_high",
    "percent_change",
    "percent_ci_low",
    "percent_ci_high",
    "condition_omnibus_p",
    "interaction_omnibus_p",
    "overall_contrast_p",
    "preregistered_interpretation"
  )

  write_csv_clean(
    paper[keep],
    file.path(
      output_dir,
      "tables",
      "PAPER_TABLE_PRIMARY_EFFECTS.csv"
    )
  )
}


save_overall_h1_h4_effect_figure <- function(summary_df, output_dir) {
  h1_h4_order <- c(
    "H1_output_tokens",
    "H2_energy",
    "H3_carbon",
    "H4_latency"
  )

  labels <- c(
    "Output tokens",
    "Energy",
    "Carbon",
    "Latency"
  )

  d <- summary_df[
    match(h1_h4_order, summary_df$analysis),
  ]

  if (anyNA(d$analysis)) {
    stop(
      "Cannot create H1-H4 summary figure because one or more H1-H4 models are missing.",
      call. = FALSE
    )
  }

  figure_dir <- file.path(output_dir, "figures")
  ensure_dir(figure_dir)

  draw_plot <- function() {
    y <- rev(seq_along(labels))
    effect <- d$percent_change
    low <- d$percent_ci_low
    high <- d$percent_ci_high

    x_range <- range(c(low, high, 0), na.rm = TRUE)

    plot(
      effect,
      y,
      xlim = x_range,
      ylim = c(0.5, length(labels) + 0.5),
      yaxt = "n",
      xlab = "Sustainability vs General change (%)",
      ylab = "",
      pch = 16,
      main = "Overall estimated effects on resource outcomes"
    )

    axis(
      2,
      at = y,
      labels = labels,
      las = 1
    )

    abline(v = 0, lty = 2)

    arrows(
      low,
      y,
      high,
      y,
      angle = 90,
      code = 3,
      length = 0.05
    )
  }

  png(
    file.path(
      figure_dir,
      "Figure_primary_H1_H4_percent_change.png"
    ),
    width = 1800,
    height = 1200,
    res = 250
  )
  par(mar = c(5, 8, 4, 2) + 0.1)
  draw_plot()
  dev.off()

  pdf(
    file.path(
      figure_dir,
      "Figure_primary_H1_H4_percent_change.pdf"
    ),
    width = 8,
    height = 5
  )
  par(mar = c(5, 8, 4, 2) + 0.1)
  draw_plot()
  dev.off()
}


write_analysis_metadata <- function(
  output_dir,
  judges_used,
  judges_complete
) {
  log_dir <- file.path(output_dir, "logs")
  ensure_dir(log_dir)

  metadata <- c(
    paste("alpha:", ALPHA),
    "tests: two-sided",
    paste("expected_prompts:", EXPECTED_PROMPTS),
    paste("expected_completed_runs:", EXPECTED_RUNS),
    paste("generation_cap_num_predict:", GENERATION_CAP),
    paste("model_slugs:", paste(EXPECTED_MODELS, collapse = ", ")),
    "model_formula: outcome ~ condition * model + (1 | prompt_id)",
    "LMM_df_method_for_emmeans: satterthwaite",
    paste(
      "followup_adjustment:",
      "Holm across the three model-specific Sustainability-vs-General contrasts",
      "only after a significant condition:model interaction"
    ),
    "cross_hypothesis_adjustment: none across prespecified H1-H5 outcome models",
    paste(
      "H1_definition:",
      "total generated output tokens reported by Ollama eval_count;",
      "may include thinking/reasoning tokens"
    ),
    paste(
      "generation_failure_handling:",
      "completed generation-limit outcomes remain in H1-H4;"
    ),
    paste(
      "empty_final_response_QC:",
      "final_content_empty is summarized regardless of done_reason"
    ),
    paste(
      "H5_empty_response_rule:",
      "empty final responses are deterministic task failures:",
      "QA=incorrect (0); scored tasks=minimum score (1);",
      "judge rows for empty responses are ignored"
    ),
    paste(
      "H5_judge_completeness_rule:",
      "every non-empty response must have exactly one valid outcome",
      "from each of the three prespecified judges"
    ),
    paste(
      "judge_models:",
      paste(EXPECTED_JUDGES, collapse = ", ")
    ),
    paste("judges_supplied:", judges_used),
    paste("H5_complete_and_run:", judges_complete),
    paste("R_version:", R.version.string),
    paste("lme4_version:", as.character(utils::packageVersion("lme4"))),
    paste("lmerTest_version:", as.character(utils::packageVersion("lmerTest"))),
    paste("emmeans_version:", as.character(utils::packageVersion("emmeans")))
  )

  writeLines(
    metadata,
    file.path(log_dir, "analysis_settings.txt")
  )

  writeLines(
    capture.output(sessionInfo()),
    file.path(log_dir, "sessionInfo.txt")
  )
}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main <- function() {
  args <- parse_args(
    commandArgs(trailingOnly = TRUE)
  )

  output_dir <- args$output_dir

  for (subdir in c(
    "tables",
    "figures",
    "logs",
    "diagnostics"
  )) {
    ensure_dir(
      file.path(output_dir, subdir)
    )
  }

  cat("\n")
  cat("Loading and validating experiment data...\n")

  exp <- prepare_experiment(
    args$experiment
  )

  cat("Experiment data valid: 600 completed measured runs.\n")

  write_csv_clean(
    exp,
    file.path(
      output_dir,
      "tables",
      "experiment_analysis_ready_H1_H4.csv"
    )
  )

  write_h1_h4_descriptives(
    exp,
    output_dir
  )

  qc <- write_generation_qc(
    exp,
    output_dir
  )

  save_generation_qc_figure(
    qc,
    output_dir
  )

  summaries <- list()

  h1_h4_specs <- list(
    list(
      label = "H1_output_tokens",
      outcome = "log_token_count"
    ),
    list(
      label = "H2_energy",
      outcome = "log_energy"
    ),
    list(
      label = "H3_carbon",
      outcome = "log_carbon"
    ),
    list(
      label = "H4_latency",
      outcome = "log_latency"
    )
  )

  cat("\nRunning H1-H4 mixed models...\n")

  for (spec in h1_h4_specs) {
    cat("  ", spec$label, "...\n", sep = "")

    summaries[[length(summaries) + 1L]] <- fit_h1_h4_model(
      exp = exp,
      outcome = spec$outcome,
      label = spec$label,
      output_dir = output_dir
    )
  }

  h5_complete <- FALSE
  full_data <- NULL

  if (!is.null(args$judges)) {
    cat("\nJudge file supplied. Validating H5 data...\n")

    judge_result <- tryCatch(
      {
        data <- prepare_judges(
          args$judges,
          exp,
          output_dir
        )

        list(
          ok = TRUE,
          data = data,
          error = NULL
        )
      },
      error = function(e) {
        list(
          ok = FALSE,
          data = NULL,
          error = conditionMessage(e)
        )
      }
    )

    if (judge_result$ok) {
      full_data <- judge_result$data
      h5_complete <- TRUE

      cat("Judge data valid and complete. Running H5...\n")

      write_csv_clean(
        full_data,
        file.path(
          output_dir,
          "tables",
          "analysis_ready_data.csv"
        )
      )

      write_h5_descriptives(
        full_data,
        output_dir
      )

      summaries[[length(summaries) + 1L]] <- fit_h5_qa(
        full_data,
        output_dir
      )

      summaries[[length(summaries) + 1L]] <- fit_h5_score(
        full_data,
        output_dir
      )

    } else {
      message <- paste0(
        "H5 was NOT run because required non-empty-response judge data ",
        "are incomplete/invalid.\n",
        judge_result$error,
        "\nH1-H4 results remain valid and were still produced. ",
        "Empty final responses do not require judge calls."
      )

      writeLines(
        message,
        file.path(
          output_dir,
          "logs",
          "H5_SKIPPED.txt"
        )
      )

      cat("\n", message, "\n", sep = "")

      if (args$require_complete_judges) {
        stop(
          paste0(
            "Complete judges were required for this run. ",
            "See logs/H5_SKIPPED.txt."
          ),
          call. = FALSE
        )
      }
    }

  } else {
    message <- paste0(
      "No --judges file was supplied. ",
      "H1-H4 were run normally; H5 was intentionally skipped."
    )

    writeLines(
      message,
      file.path(
        output_dir,
        "logs",
        "H5_SKIPPED.txt"
      )
    )

    cat("\n", message, "\n", sep = "")

    if (args$require_complete_judges) {
      stop(
        "--require-complete-judges was set but no --judges file was supplied.",
        call. = FALSE
      )
    }
  }

  summary_df <- do.call(
    rbind,
    summaries
  )

  summary_df <- add_interpretation(
    summary_df
  )

  write_csv_clean(
    summary_df,
    file.path(
      output_dir,
      "tables",
      "FINAL_ANALYSIS_SUMMARY.csv"
    )
  )

  write_paper_primary_table(
    summary_df,
    output_dir
  )

  save_overall_h1_h4_effect_figure(
    summary_df,
    output_dir
  )

  write_analysis_metadata(
    output_dir = output_dir,
    judges_used = !is.null(args$judges),
    judges_complete = h5_complete
  )

  cat("\nAnalysis complete.\n")
  cat(
    "Primary table: ",
    file.path(
      output_dir,
      "tables",
      "PAPER_TABLE_PRIMARY_EFFECTS.csv"
    ),
    "\n",
    sep = ""
  )
  cat(
    "H1-H4 figure: ",
    file.path(
      output_dir,
      "figures",
      "Figure_primary_H1_H4_percent_change.png"
    ),
    "\n",
    sep = ""
  )

  if (!h5_complete) {
    cat(
      "H5 status: skipped. See ",
      file.path(
        output_dir,
        "logs",
        "H5_SKIPPED.txt"
      ),
      "\n",
      sep = ""
    )
  } else {
    cat(
      "H5 status: complete. Non-empty responses used the 3-judge panel; ",
      "empty responses used the deterministic failure rule.\n",
      sep = ""
    )
  }
}


main()
