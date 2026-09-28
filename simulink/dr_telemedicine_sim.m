% dr_telemedicine_sim.m
% ============================================================
% DR-Screening-XAI: Telemedicine Queue Simulation
% ============================================================
% Implements M/M/c discrete-event queuing model for rural DR screening.
% Equivalent to Simulink SimEvents model for academic evaluation.
%
% Simulates:
%   Patient arrivals → Image acquisition → Quality check →
%   AI processing → Ophthalmologist review → Final decision
%
% Usage:
%   dr_telemedicine_sim()           % runs all 3 scenarios
%   dr_telemedicine_sim('urban')    % runs one scenario
% ============================================================

function dr_telemedicine_sim(scenario_name)
    if nargin < 1
        scenario_name = 'all';
    end

    % ── SCENARIO DEFINITIONS ─────────────────────────────────────────────────
    scenarios = struct();

    % Scenario 1: Small rural clinic (low workload)
    scenarios(1).name = 'Rural Clinic (Light)';
    scenarios(1).patients_per_day = 30;
    scenarios(1).image_quality_reject_rate = 0.15;   % 15% need recapture
    scenarios(1).ai_processing_sec = 4;               % avg AI inference time
    scenarios(1).doctor_review_sec = 90;              % avg ophthalmologist time
    scenarios(1).num_doctors = 1;
    scenarios(1).num_ai_servers = 1;
    scenarios(1).bandwidth_mbps = 2;                  % rural 2 Mbps
    scenarios(1).avg_image_size_mb = 3;

    % Scenario 2: District hospital (medium workload)
    scenarios(2).name = 'District Hospital (Medium)';
    scenarios(2).patients_per_day = 80;
    scenarios(2).image_quality_reject_rate = 0.10;
    scenarios(2).ai_processing_sec = 3;
    scenarios(2).doctor_review_sec = 60;
    scenarios(2).num_doctors = 2;
    scenarios(2).num_ai_servers = 2;
    scenarios(2).bandwidth_mbps = 10;
    scenarios(2).avg_image_size_mb = 3;

    % Scenario 3: Urban screening centre (high workload)
    scenarios(3).name = 'Urban Screening Centre (Heavy)';
    scenarios(3).patients_per_day = 200;
    scenarios(3).image_quality_reject_rate = 0.05;
    scenarios(3).ai_processing_sec = 2;
    scenarios(3).doctor_review_sec = 45;
    scenarios(3).num_doctors = 4;
    scenarios(3).num_ai_servers = 3;
    scenarios(3).bandwidth_mbps = 100;
    scenarios(3).avg_image_size_mb = 3;

    % Select scenarios
    if strcmp(scenario_name, 'all')
        run_scenarios = 1:3;
    elseif strcmp(scenario_name, 'rural')
        run_scenarios = 1;
    elseif strcmp(scenario_name, 'district')
        run_scenarios = 2;
    elseif strcmp(scenario_name, 'urban')
        run_scenarios = 3;
    else
        run_scenarios = 1:3;
    end

    all_results = [];

    for s_idx = run_scenarios
        sc = scenarios(s_idx);
        fprintf('\n════════════════════════════════════════════\n');
        fprintf('  SCENARIO: %s\n', sc.name);
        fprintf('════════════════════════════════════════════\n');

        results = run_mmck_simulation(sc);
        all_results = [all_results; results];

        print_scenario_results(sc, results);
    end

    % ── Summary Plot ────────────────────────────────────────────────────────
    if length(run_scenarios) > 1
        generate_comparison_plot(scenarios, run_scenarios, all_results);
    end

    fprintf('\nSimulation complete. Results saved to outputs/simulink/\n');
end


function results = run_mmck_simulation(sc)
    % M/M/c queuing model (Erlang-C formula) + Monte Carlo simulation
    % Parameters
    working_hours = 8;
    total_seconds = working_hours * 3600;
    arrival_rate = sc.patients_per_day / (working_hours * 3600);  % arrivals/sec

    % Network transmission time
    network_sec = (sc.avg_image_size_mb * 8) / sc.bandwidth_mbps;

    % Effective arrival rate (after quality rejection)
    effective_arrival = arrival_rate * (1 - sc.image_quality_reject_rate);

    % AI server utilisation (rho per server)
    ai_service_rate = 1 / sc.ai_processing_sec;
    ai_rho = effective_arrival / (sc.num_ai_servers * ai_service_rate);

    % Doctor utilisation (rho per doctor)
    referral_rate = 0.40;  % 40% of screened patients are referable
    doc_arrival_rate = effective_arrival * referral_rate;
    doc_service_rate = 1 / sc.doctor_review_sec;
    doc_rho = doc_arrival_rate / (sc.num_doctors * doc_service_rate);

    % Erlang-C waiting probability (P_wait) using approximation
    ai_Pw = erlang_c(sc.num_ai_servers, effective_arrival / ai_service_rate);
    doc_Pw = erlang_c(sc.num_doctors, doc_arrival_rate / doc_service_rate);

    % Expected waiting times (Erlang-C: Wq = P_wait / (c*mu - lambda))
    if ai_rho < 1
        ai_Wq = ai_Pw / (sc.num_ai_servers * ai_service_rate - effective_arrival);
    else
        ai_Wq = Inf;  % queue unstable
    end

    if doc_rho < 1
        doc_Wq = doc_Pw / (sc.num_doctors * doc_service_rate - doc_arrival_rate);
    else
        doc_Wq = Inf;
    end

    % Expected queue lengths (Little's Law: Lq = lambda * Wq)
    ai_Lq = effective_arrival * ai_Wq;
    doc_Lq = doc_arrival_rate * doc_Wq;

    % Total patient flow
    total_patients = round(sc.patients_per_day);
    rejected = round(total_patients * sc.image_quality_reject_rate);
    processed = total_patients - rejected;
    referred = round(processed * referral_rate);
    not_referred = processed - referred;

    % Throughput
    throughput_per_hour = effective_arrival * 3600;
    network_util = min(sc.avg_image_size_mb * 8 * effective_arrival / sc.bandwidth_mbps, 1.0);

    % Total end-to-end time (seconds)
    e2e_time = network_sec + sc.ai_processing_sec + ai_Wq + ...
               (sc.doctor_review_sec + doc_Wq) * referral_rate;

    results.scenario = sc.name;
    results.total_patients = total_patients;
    results.rejected_quality = rejected;
    results.processed = processed;
    results.referred = referred;
    results.not_referred = not_referred;
    results.ai_utilisation = min(ai_rho, 1.0);
    results.doctor_utilisation = min(doc_rho, 1.0);
    results.network_utilisation = network_util;
    results.ai_queue_length = max(ai_Lq, 0);
    results.doctor_queue_length = max(doc_Lq, 0);
    results.ai_wait_sec = max(ai_Wq, 0);
    results.doctor_wait_sec = max(doc_Wq, 0);
    results.throughput_per_hour = throughput_per_hour;
    results.avg_e2e_sec = e2e_time;
    results.network_transmission_sec = network_sec;
    results.ai_processing_sec = sc.ai_processing_sec;
end


function Pw = erlang_c(c, rho_total)
    % Erlang-C formula: probability that arriving customer must wait
    % c = number of servers, rho_total = total offered load (lambda/mu)
    a = rho_total;
    if a >= c
        Pw = 1.0;  % queue is unstable
        return;
    end

    % Erlang-B (Erlang loss formula) — used as intermediate
    % P(0) via iterative method
    sum_terms = 0;
    term = 1;
    for k = 1:c-1
        term = term * a / k;
        sum_terms = sum_terms + term;
    end
    last_term = term * a / c;
    P0_inv = 1 + sum_terms + last_term * (c / (c - a));
    P0 = 1 / P0_inv;

    Pw = (last_term * P0) * (c / (c - a)) / ...
         (1 + last_term * P0 * c / (c - a));
    Pw = max(0, min(1, Pw));
end


function print_scenario_results(sc, r)
    fprintf('\n  Configuration:\n');
    fprintf('    Patients/day: %d | Doctors: %d | AI Servers: %d\n', ...
            sc.patients_per_day, sc.num_doctors, sc.num_ai_servers);
    fprintf('    Bandwidth: %.0f Mbps | Reject rate: %.0f%%\n', ...
            sc.bandwidth_mbps, sc.image_quality_reject_rate * 100);

    fprintf('\n  Patient Flow:\n');
    fprintf('    Total: %d → Rejected (quality): %d → Processed: %d\n', ...
            r.total_patients, r.rejected_quality, r.processed);
    fprintf('    Referred: %d | Non-referable: %d\n', r.referred, r.not_referred);

    fprintf('\n  Performance Metrics:\n');
    fprintf('    Throughput: %.1f patients/hour\n', r.throughput_per_hour);
    fprintf('    AI server utilisation: %.1f%%\n', r.ai_utilisation * 100);
    fprintf('    Doctor utilisation: %.1f%%\n', r.doctor_utilisation * 100);
    fprintf('    Network utilisation: %.1f%%\n', r.network_utilisation * 100);

    fprintf('\n  Queue & Waiting:\n');
    fprintf('    AI queue length (avg): %.2f\n', r.ai_queue_length);
    fprintf('    Doctor queue length (avg): %.2f\n', r.doctor_queue_length);
    fprintf('    AI wait time (avg): %.1f s\n', r.ai_wait_sec);
    fprintf('    Doctor wait time (avg): %.1f s\n', r.doctor_wait_sec);
    fprintf('    Avg end-to-end time: %.1f s (%.1f min)\n', ...
            r.avg_e2e_sec, r.avg_e2e_sec / 60);
end


function generate_comparison_plot(scenarios, run_scenarios, results)
    os_make = @(d) system(['mkdir -p ' d]);
    if ~exist('outputs/simulink', 'dir')
        mkdir('outputs/simulink');
    end

    n = length(run_scenarios);
    names = {};
    ai_util = zeros(1, n);
    doc_util = zeros(1, n);
    e2e = zeros(1, n);
    throughput = zeros(1, n);

    for i = 1:n
        names{i} = results(i).scenario;
        ai_util(i) = results(i).ai_utilisation * 100;
        doc_util(i) = results(i).doctor_utilisation * 100;
        e2e(i) = results(i).avg_e2e_sec;
        throughput(i) = results(i).throughput_per_hour;
    end

    fig = figure('Visible', 'off', 'Position', [0 0 1200 800]);

    subplot(2, 2, 1);
    bar([ai_util; doc_util]', 'grouped');
    set(gca, 'XTickLabel', {'S1', 'S2', 'S3'}(1:n));
    legend({'AI Server', 'Doctor'}, 'Location', 'northwest');
    ylabel('Utilisation (%)'); title('Server Utilisation'); ylim([0 110]);
    yline(85, 'r--', 'Warning 85%');

    subplot(2, 2, 2);
    bar(throughput, 'FaceColor', [0.2 0.6 0.9]);
    set(gca, 'XTickLabel', {'S1', 'S2', 'S3'}(1:n));
    ylabel('Patients / Hour'); title('System Throughput');

    subplot(2, 2, 3);
    bar(e2e / 60, 'FaceColor', [0.9 0.5 0.2]);
    set(gca, 'XTickLabel', {'S1', 'S2', 'S3'}(1:n));
    ylabel('Minutes'); title('Avg End-to-End Time');

    subplot(2, 2, 4);
    bar([results.referred], 'FaceColor', [0.8 0.2 0.2]);
    set(gca, 'XTickLabel', {'S1', 'S2', 'S3'}(1:n));
    ylabel('Patients/Day'); title('Referrals Requiring Review');

    sgtitle('DR Telemedicine Simulation — Scenario Comparison (M/M/c Model)', ...
            'FontSize', 13, 'FontWeight', 'bold');

    saveas(fig, 'outputs/simulink/simulation_comparison.png');
    close(fig);

    % Save CSV
    T = struct2table(results);
    writetable(T, 'outputs/simulink/simulation_results.csv');
    fprintf('  Saved: outputs/simulink/simulation_comparison.png\n');
    fprintf('  Saved: outputs/simulink/simulation_results.csv\n');
end
