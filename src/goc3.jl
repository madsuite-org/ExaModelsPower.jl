function goc3_model(
    filename, uc_filename;
    backend = nothing,
    T = Float64,
    include_ctg = true, #contingencies will be specified in input data
    result_set = [],
    user_callback = dummy_extension,
    kwargs...
        )


    static = goc3_static(filename)
    sc_data, lengths = parse_sc_data(static, goc3_commitment(uc_filename, static))
    @info "parsed data"

    (L_J_xf, L_J_ln, L_J_ac, L_J_dc, L_J_br, L_J_cs,
    L_J_pr, L_J_cspr, L_J_sh, I, L_T, L_N_p, L_N_q, L_W_en_min_pr,
     L_W_en_min_cs, L_W_en_max_pr, L_W_en_max_cs, K) = lengths

    # The uids stay on the host, beside the arrays the model is built from: they
    # are what `save_go3_solution` names each position by.
    sc_data_array = (; sc_data..., uids = static.uids)
    sc_data = convert_data(sc_data, backend)

    core = ExaCore(T; backend = backend)

    if result_set != []
        initialize_vars = true
        result = result_set[1]
        vars = result_set[2]
    else
        initialize_vars = false
    end

    #variables are indexed j,t,k or j,t (t always second if present)
    @add_var(core, b_jt_sh, L_J_sh, L_T; start = initialize_vars ? Array(solution(result, vars.b_jt_sh)) : 1)
    @add_var(core, g_jt_sh, L_J_sh, L_T; start = initialize_vars ? Array(solution(result, vars.g_jt_sh)) : 1)
    #Split e_w_plus into separate sets for W_en_min and W_en_max and for pr, cs
    #Bounds from 4.6.3 Maximum/minimum energy over multiple intervals (77)
    @add_var(core, e_w_plus_min_pr, L_W_en_min_pr; lvar = 0, start = initialize_vars ? Array(solution(result, vars.e_w_plus_min_pr)) : 0)
    @add_var(core, e_w_plus_min_cs, L_W_en_min_cs; lvar = 0, start = initialize_vars ? Array(solution(result, vars.e_w_plus_min_cs)) : 0)
    @add_var(core, e_w_plus_max_pr, L_W_en_max_pr; lvar = 0, start = initialize_vars ? Array(solution(result, vars.e_w_plus_max_pr)) : 0)
    @add_var(core, e_w_plus_max_cs, L_W_en_max_cs; lvar = 0, start = initialize_vars ? Array(solution(result, vars.e_w_plus_max_cs)) : 0)
    @add_var(core, p_it, I, L_T; start = initialize_vars ? Array(solution(result, vars.p_it)) : 1)
    @add_var(core, p_it_plus, I, L_T; start = initialize_vars ? Array(solution(result, vars.p_it_plus)) : 1)
    #splitting p_jt and q_jt for shunts, producers, and consumers
    @add_var(core, p_jt_sh, L_J_sh, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_sh)) : 1)
    @add_var(core, p_jt_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_pr)) : 1)
    @add_var(core, p_jt_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_cs)) : 1)
    @add_var(core, q_jt_sh, L_J_sh, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_sh)) : 1)
    @add_var(core, q_jt_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_pr)) : 1)
    @add_var(core, q_jt_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_cs)) : 1)
    #Splitting p on, sd , su into pr and cs
    @add_var(core, p_jt_on_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_on_pr)) : 1)
    @add_var(core, p_jt_on_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_on_cs)) : 1)
    @add_var(core, p_jt_su_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_su_pr)) : 1)
    @add_var(core, p_jt_su_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_su_cs)) : 1)
    @add_var(core, p_jt_sd_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_sd_pr)) : 1)
    @add_var(core, p_jt_sd_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_sd_cs)) : 1)
    #p_jtm has been flattened and uses only one special index, k_flat
    #Bounds from 4.6.9 Energy cost and value (129)
    @add_var(core, p_jtm_pr, length(sc_data.p_jtm_flattened_pr); lvar = 0, uvar = sc_data.p_jtm_pr_uvar, start = initialize_vars ? Array(solution(result, vars.p_jtm_pr)) : 1)
    @add_var(core, p_jtm_cs, length(sc_data.p_jtm_flattened_cs); lvar = 0, uvar = sc_data.p_jtm_cs_uvar, start = initialize_vars ? Array(solution(result, vars.p_jtm_cs)) : 1)
    #to/from power split into ln, xf, and dc lines
    #Bounds from 4.8.4 DC lines (152-155)
    @add_var(core, p_jt_fr_ln, L_J_ln, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_fr_ln)) : 1)
    @add_var(core, p_jt_fr_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_fr_xf)) : 1)
    @add_var(core, p_jt_fr_dc, L_J_dc, L_T; lvar = -sc_data.p_jt_fr_dc_max, uvar = sc_data.p_jt_fr_dc_max, start = initialize_vars ? Array(solution(result, vars.p_jt_fr_dc)) : 1)
    @add_var(core, p_jt_to_ln, L_J_ln, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_to_ln)) : 1)
    @add_var(core, p_jt_to_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.p_jt_to_xf)) : 1)
    @add_var(core, p_jt_to_dc, L_J_dc, L_T; lvar = -sc_data.p_jt_to_dc_max, uvar = sc_data.p_jt_to_dc_max, start = initialize_vars ? Array(solution(result, vars.p_jt_to_dc)) : 1)
    @add_var(core, q_jt_fr_ln, L_J_ln, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_fr_ln)) : 1)
    @add_var(core, q_jt_fr_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_fr_xf)) : 1)
    @add_var(core, q_jt_fr_dc, L_J_dc, L_T; lvar = sc_data.q_jt_fr_dc_lvar, uvar = sc_data.q_jt_fr_dc_uvar, start = initialize_vars ? Array(solution(result, vars.q_jt_fr_dc)) : 1)
    @add_var(core, q_jt_to_ln, L_J_ln, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_to_ln)) : 1)
    @add_var(core, q_jt_to_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.q_jt_to_xf)) : 1)
    @add_var(core, q_jt_to_dc, L_J_dc, L_T; lvar = sc_data.q_jt_to_dc_lvar, uvar = sc_data.q_jt_to_dc_uvar, start = initialize_vars ? Array(solution(result, vars.q_jt_to_dc)) : 1)
    #p_jt rgu, rgd, scr, rru,on, rru,off, rrd,on, rrd,off and q_jt qru/qrd split into pr and cs
    #bounds from 4.6.4 Device reserve variable domains (80-89)
    @add_var(core, p_jt_rgu_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rgu_pr)) : 1)
    @add_var(core, p_jt_rgu_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rgu_cs)) : 1)
    @add_var(core, p_jt_rgd_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rgd_pr)) : 1)
    @add_var(core, p_jt_rgd_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rgd_cs)) : 1)
    @add_var(core, p_jt_scr_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_scr_pr)) : 1)
    @add_var(core, p_jt_scr_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_scr_cs)) : 1)
    @add_var(core, p_jt_nsc_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_nsc_pr)) : 1)
    @add_var(core, p_jt_rru_on_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rru_on_pr)) : 1)
    @add_var(core, p_jt_rru_on_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rru_on_cs)) : 1)
    @add_var(core, p_jt_rru_off_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rru_off_pr)) : 1)
    @add_var(core, p_jt_rrd_on_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rrd_on_pr)) : 1)
    @add_var(core, p_jt_rrd_on_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rrd_on_cs)) : 1)
    @add_var(core, p_jt_rrd_off_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_jt_rrd_off_cs)) : 1)
    @add_var(core, q_jt_qru_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_jt_qru_pr)) : 1)
    @add_var(core, q_jt_qru_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_jt_qru_cs)) : 1)
    @add_var(core, q_jt_qrd_pr, L_J_pr, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_jt_qrd_pr)) : 1)
    @add_var(core, q_jt_qrd_cs, L_J_cs, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_jt_qrd_cs)) : 1)

    
    @add_var(core, p_nt_rgu_req, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.p_nt_rgu_req)) : 1)
    @add_var(core, p_nt_rgd_req, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.p_nt_rgd_req)) : 1)
    @add_var(core, p_nt_scr_req, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.p_nt_scr_req)) : 1)
    @add_var(core, p_nt_nsc_req, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.p_nt_nsc_req)) : 1)

    @add_var(core, p_jt_pr_max, L_N_p, L_T;)
    #Bounds from 4.3.1 Reserve shortfall domains (20-27)
    @add_var(core, p_nt_rgu_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_rgu_plus)) : 1)
    @add_var(core, p_nt_rgd_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_rgd_plus)) : 1)
    @add_var(core, p_nt_scr_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_scr_plus)) : 1)
    @add_var(core, p_nt_nsc_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_nsc_plus)) : 1)
    @add_var(core, p_nt_rru_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_rru_plus)) : 1)
    @add_var(core, p_nt_rrd_plus, L_N_p, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.p_nt_rrd_plus)) : 1)
    @add_var(core, q_nt_qru_plus, L_N_q, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_nt_qru_plus)) : 1)
    @add_var(core, q_nt_qrd_plus, L_N_q, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.q_nt_qrd_plus)) : 1)

    @add_var(core, q_it, I, L_T; start = initialize_vars ? Array(solution(result, vars.q_it)) : 1)
    @add_var(core, q_it_plus, I, L_T; start = initialize_vars ? Array(solution(result, vars.q_it_plus)) : 1)

    #s_jt_plus split on ln and xf
    #Bounds from 4.8.1 AC branch flow limits and penalties (138)
    @add_var(core, s_jt_plus_ln, L_J_ln, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.s_jt_plus_ln)) : 1)
    @add_var(core, s_jt_plus_xf, L_J_xf, L_T; lvar = 0, start = initialize_vars ? Array(solution(result, vars.s_jt_plus_xf)) : 1)

    #Bounds form 4.2.4 Bus voltage (19)
    
    @add_var(core, v_it, I, L_T; lvar = sc_data.v_lvar, uvar = sc_data.v_uvar, start = initialize_vars ? Array(solution(result, vars.v_it)) : ones(I, L_T)) 

    @add_var(core, z_w_en_max_pr, L_W_en_max_pr; start = initialize_vars ? Array(solution(result, vars.z_w_en_max_pr)) : 0)
    @add_var(core, z_w_en_max_cs, L_W_en_max_cs; start = initialize_vars ? Array(solution(result, vars.z_w_en_max_cs)) : 0)
    @add_var(core, z_w_en_min_pr, L_W_en_min_pr; start = initialize_vars ? Array(solution(result, vars.z_w_en_min_pr)) : 0)
    @add_var(core, z_w_en_min_cs, L_W_en_min_cs; start = initialize_vars ? Array(solution(result, vars.z_w_en_min_cs)) : 0)

    #split z_jt_en and on into pr and cs
    @add_var(core, z_jt_en_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_en_pr)) : 0)
    @add_var(core, z_jt_en_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_en_cs)) : 0)

    @add_var(core, z_it_p, I, L_T; start = initialize_vars ? Array(solution(result, vars.z_it_p)) : 0)
    @add_var(core, z_it_q, I, L_T; start = initialize_vars ? Array(solution(result, vars.z_it_q)) : 0)

    #z_jt_s split into ln and xf
    @add_var(core, z_jt_s_ln, L_J_ln, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_s_ln)) : 0)
    @add_var(core, z_jt_s_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_s_xf)) : 0)
    #z_jt rgu, rgd, scr, nsc, rru, rrd, qru, qrd split into pr and cs
    @add_var(core, z_jt_rgu_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rgu_pr)) : 0)
    @add_var(core, z_jt_rgu_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rgu_cs)) : 0)
    @add_var(core, z_jt_rgd_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rgd_pr)) : 0)
    @add_var(core, z_jt_rgd_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rgd_cs)) : 0)
    @add_var(core, z_jt_scr_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_scr_pr)) : 0)
    @add_var(core, z_jt_scr_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_scr_cs)) : 0)
    @add_var(core, z_jt_nsc_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_nsc_pr)) : 0)
    @add_var(core, z_jt_rru_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rru_pr)) : 0)
    @add_var(core, z_jt_rru_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rru_cs)) : 0)
    @add_var(core, z_jt_rrd_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rrd_pr)) : 0)
    @add_var(core, z_jt_rrd_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_rrd_cs)) : 0)
    @add_var(core, z_jt_qru_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_qru_pr)) : 0)
    @add_var(core, z_jt_qru_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_qru_cs)) : 0)
    @add_var(core, z_jt_qrd_pr, L_J_pr, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_qrd_pr)) : 0)
    @add_var(core, z_jt_qrd_cs, L_J_cs, L_T; start = initialize_vars ? Array(solution(result, vars.z_jt_qrd_cs)) : 0)

    @add_var(core, z_nt_rgu, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_rgu)) : 0)
    @add_var(core, z_nt_rgd, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_rgd)) : 0)
    @add_var(core, z_nt_scr, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_scr)) : 0)
    @add_var(core, z_nt_nsc, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_nsc)) : 0)
    @add_var(core, z_nt_rru, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_rru)) : 0)
    @add_var(core, z_nt_rrd, L_N_p, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_rrd)) : 0)
    @add_var(core, z_nt_qru, L_N_q, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_qru)) : 0)
    @add_var(core, z_nt_qrd, L_N_q, L_T; start = initialize_vars ? Array(solution(result, vars.z_nt_qru)) : 0)

    #Bounds added so that angle doesnt blow up and cause computational errors
    @add_var(core, θ_it, I, L_T; lvar = 0, uvar = pi + 0.0001, start = initialize_vars ? Array(solution(result, vars.θ_it)) : 0)

    #split τjt and φjt into xf only, ln is fixed
    @add_var(core, τ_jt_xf, L_J_xf, L_T; start = initialize_vars ? Array(solution(result, vars.τ_jt_xf)) : 1)
    #Bounds added so that angle doesnt blow up and cause computational errors
    @add_var(core, φ_jt_xf, L_J_xf, L_T; lvar = 0, uvar = pi + 0.0001, start = initialize_vars ? Array(solution(result, vars.φ_jt_xf)) : 0)

    
    if include_ctg
        #Bound from 4.9.1 Penalty on post-contingency AC branch overload (157)
        @add_var(core, s_jtk_plus_ln, length(sc_data.jtk_ln_flattened); lvar = 0, start = initialize_vars ? Array(solution(result, vars.s_jtk_plus_ln)) : 0)
        @add_var(core, s_jtk_plus_xf, length(sc_data.jtk_xf_flattened); lvar = 0, start = initialize_vars ? Array(solution(result, vars.s_jtk_plus_xf)) : 0)
        @add_var(core, z_jtk_s_ln, length(sc_data.jtk_ln_flattened); start = initialize_vars ? Array(solution(result, vars.z_jtk_s_ln)) : 0)
        @add_var(core, z_jtk_s_xf, length(sc_data.jtk_xf_flattened); start = initialize_vars ? Array(solution(result, vars.z_jtk_s_xf)) : 0)
        @add_var(core, p_jtk_ln, length(sc_data.jtk_ln_flattened); start = initialize_vars ? Array(solution(result, vars.p_jtk_ln)) : 1)
        @add_var(core, p_jtk_xf, length(sc_data.jtk_xf_flattened); start = initialize_vars ? Array(solution(result, vars.p_jtk_xf)) : 1)
        @add_var(core, θ_itk, I, L_T, K; start = initialize_vars ? Array(solution(result, vars.θ_itk)) : 0)
        @add_var(core, p_t_sl, L_T; start = initialize_vars ? Array(solution(result, vars.p_t_sl)) : 1)
        @add_var(core, z_tk_ctg, L_T, K; start = initialize_vars ? Array(solution(result, vars.z_tk_ctg)) : 0)
        @add_var(core, z_t_ctg_min, L_T; start = initialize_vars ? Array(solution(result, vars.z_t_ctg_min)) : 0)
        @add_var(core, z_t_ctg_avg, L_T; start = initialize_vars ? Array(solution(result, vars.z_t_ctg_avg)) : 0)
    end
    

    #4.1 Market surplus objective
    #constraint (6-9)
    #All objectives are negative so that we can minimize

    #Removing all uc variables, which include z_on, z_su, z_sd, z_sus. Note that this means objective value from ExaModelsPower will not match the objective calculated for the full GOC3 model, even though we still are solving the same problem (with UC as constants)

    if include_ctg
        @add_obj(core, o2, -z_t_ctg_min[t] for t in sc_data.periods)
        @add_obj(core, o3, -z_t_ctg_avg[t] for t in sc_data.periods)
    end

    @add_obj(core, o6_t_pr, -(-z_jt_en_pr[pr.j_pr, pr.t] 
                        - (z_jt_rgu_pr[pr.j_pr, pr.t] + z_jt_rgd_pr[pr.j_pr, pr.t] + z_jt_scr_pr[pr.j_pr, pr.t] + z_jt_nsc_pr[pr.j_pr, pr.t] + z_jt_rru_pr[pr.j_pr, pr.t] + 
                        z_jt_rrd_pr[pr.j_pr, pr.t] + z_jt_qru_pr[pr.j_pr, pr.t] +z_jt_qrd_pr[pr.j_pr, pr.t])) for pr in sc_data.prarray)
    @add_obj(core, o6_t_cs, -(z_jt_en_cs[cs.j_cs, cs.t] 
                        - (z_jt_rgu_cs[cs.j_cs, cs.t] + z_jt_rgd_cs[cs.j_cs, cs.t] + z_jt_scr_cs[cs.j_cs, cs.t] + z_jt_rru_cs[cs.j_cs, cs.t] + 
                        z_jt_rrd_cs[cs.j_cs, cs.t] + z_jt_qru_cs[cs.j_cs, cs.t] +z_jt_qrd_cs[cs.j_cs, cs.t])) for cs in sc_data.csarray)
    @add_obj(core, o6_t_ln, -(- z_jt_s_ln[ln.j_ln, ln.t]) for ln in sc_data.aclbrancharray)
    @add_obj(core, o6_t_xf, -(- z_jt_s_xf[xf.j_xf, xf.t]) for xf in sc_data.acxbrancharray)
    @add_obj(core, o6_t_i, -(-(z_it_p[b.i, b.t] + z_it_q[b.i, b.t])) for b in sc_data.busarray)
    @add_obj(core, o6_t_Np, -(-(z_nt_rgu[n.n_p, n.t] + z_nt_rgd[n.n_p, n.t] + z_nt_scr[n.n_p, n.t] + z_nt_nsc[n.n_p, n.t] + z_nt_rru[n.n_p, n.t] + z_nt_rrd[n.n_p, n.t])) for n in sc_data.preservearray)
    @add_obj(core, o6_t_Nq, -(-(z_nt_qru[n.n_q, n.t] + z_nt_qrd[n.n_q, n.t])) for n in sc_data.qreservearray)
    if L_W_en_max_pr > 0
        @add_obj(core, o6_en_max_pr, z_w_en_max_pr[w] for w in 1:L_W_en_max_pr)
    end
    if L_W_en_max_cs > 0
        @add_obj(core, o6_en_max_cs, z_w_en_max_cs[w] for w in 1:L_W_en_max_cs)
    end
    if L_W_en_min_pr > 0
        @add_obj(core, o6_en_min_pr, z_w_en_min_pr[w] for w in 1:L_W_en_min_pr)
    end
    if L_W_en_min_cs > 0
        @add_obj(core, o6_en_min_cs, z_w_en_min_cs[w] for w in 1:L_W_en_min_cs)
    end


    
    if include_ctg
        if K>0
            @add_con(core, c4, z_t_ctg_min[ind.t] - z_tk_ctg[ind.t, ind.k] for ind in sc_data.tk_index; lcon = fill(-Inf, size(sc_data.tk_index)))
            @add_con(core, c5, z_t_ctg_avg[t]*K for t in sc_data.periods) 
            @add_con!(core, c5, ind.t => -z_tk_ctg[ind.t, ind.k] for ind in sc_data.tk_index)
        else
            @add_con(core, c4, z_t_ctg_min[t] for t in sc_data.periods)
            @add_con(core, c5, z_t_ctg_avg[t] for t in sc_data.periods)
        end
        @add_con(core, c10, z_tk_ctg[ind.t, ind.k] for ind in sc_data.tk_index)
        @add_con!(core, c10, ln.t + L_T*(ln.ctg-1) => z_jtk_s_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened)
        @add_con!(core, c10, xf.t + L_T*(xf.ctg-1) => z_jtk_s_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened)
    end

    #4.2.1 Bus power mismatch and penalized mismatch definitions
    @add_con(core, c11, p_it_plus[b.i, b.t] - p_it[b.i, b.t] for b in sc_data.busarray; ucon = fill(Inf, size(sc_data.busarray)))
    @add_con(core, c12, p_it_plus[b.i, b.t] + p_it[b.i, b.t] for b in sc_data.busarray; ucon = fill(Inf, size(sc_data.busarray)))
    @add_con(core, c13, q_it_plus[b.i, b.t] - q_it[b.i, b.t] for b in sc_data.busarray; ucon = fill(Inf, size(sc_data.busarray)))
    @add_con(core, c14, q_it_plus[b.i, b.t] + q_it[b.i, b.t] for b in sc_data.busarray; ucon = fill(Inf, size(sc_data.busarray)))
    cons = (c11=c11, c12=c12, c13=c13, c14=c14)
    #4.2.2 Bus pwoer mismatch penalty
    @add_con(core, c15, z_it_p[b.i, b.t] - b.dt*sum(sc_data.c_p)*p_it_plus[b.i, b.t] for b in sc_data.busarray)
    @add_con(core, c16, z_it_q[b.i, b.t] - b.dt*sum(sc_data.c_q)*q_it_plus[b.i, b.t] for b in sc_data.busarray)
    #4.2.3 Bus real and reactive power balance
    @add_con(core, c17, p_it[b.i, b.t] for b in sc_data.busarray)
    @add_con!(core, c17, pr.bus + I*(pr.t-1) => p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con!(core, c17, cs.bus + I*(cs.t-1) => -p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con!(core, c17, sh.bus + I*(sh.t-1) => -p_jt_sh[sh.j_sh, sh.t] for sh in sc_data.shuntarray)
    
    
    #Reminder: fr and to split for ln, xf, and dc
    @add_con!(core, c17, ln.fr_bus + I*(ln.t-1) => -p_jt_fr_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray)
    @add_con!(core, c17, xf.fr_bus + I*(xf.t-1) => -p_jt_fr_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray)
    @add_con!(core, c17, dc.fr_bus + I*(dc.t-1) => -p_jt_fr_dc[dc.j_dc, dc.t] for dc in sc_data.dclinearray)
    @add_con!(core, c17, ln.to_bus + I*(ln.t-1) => -p_jt_to_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray)
    @add_con!(core, c17, xf.to_bus + I*(xf.t-1) => -p_jt_to_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray)
    @add_con!(core, c17, dc.to_bus + I*(dc.t-1) => -p_jt_to_dc[dc.j_dc, dc.t] for dc in sc_data.dclinearray)
    @add_con(core, c18, q_it[b.i, b.t] for b in sc_data.busarray)
    @add_con!(core, c18, pr.bus + I*(pr.t-1) => q_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con!(core, c18, cs.bus + I*(cs.t-1) => -q_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con!(core, c18, sh.bus + I*(sh.t-1) => -q_jt_sh[sh.j_sh, sh.t] for sh in sc_data.shuntarray)
    #Reminder: fr and to split for ln, xf, and dc
    @add_con!(core, c18, ln.fr_bus + I*(ln.t-1) => -q_jt_fr_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray)
    @add_con!(core, c18, xf.fr_bus + I*(xf.t-1) => -q_jt_fr_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray)
    @add_con!(core, c18, dc.fr_bus + I*(dc.t-1) => -q_jt_fr_dc[dc.j_dc, dc.t] for dc in sc_data.dclinearray)
    @add_con!(core, c18, ln.to_bus + I*(ln.t-1) => -q_jt_to_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray)
    @add_con!(core, c18, xf.to_bus + I*(xf.t-1) => -q_jt_to_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray)
    @add_con!(core, c18, dc.to_bus + I*(dc.t-1) => -q_jt_to_dc[dc.j_dc, dc.t] for dc in sc_data.dclinearray)
    
    #4.3.2 Reserve shortfall penalties
    @add_con(core, c28, z_nt_rgu[n.n_p, n.t] - n.dt*n.c_rgu*p_nt_rgu_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c29, z_nt_rgd[n.n_p, n.t] - n.dt*n.c_rgd*p_nt_rgd_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c30, z_nt_scr[n.n_p, n.t] - n.dt*n.c_scr*p_nt_scr_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c31, z_nt_nsc[n.n_p, n.t] - n.dt*n.c_nsc*p_nt_nsc_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c32, z_nt_rru[n.n_p, n.t] - n.dt*n.c_rru*p_nt_rru_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c33, z_nt_rrd[n.n_p, n.t] - n.dt*n.c_rrd*p_nt_rrd_plus[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c34, z_nt_qru[n.n_q, n.t] - n.dt*n.c_qru*q_nt_qru_plus[n.n_q, n.t] for n in sc_data.qreservearray)
    @add_con(core, c35, z_nt_qrd[n.n_q, n.t] - n.dt*n.c_qrd*q_nt_qrd_plus[n.n_q, n.t] for n in sc_data.qreservearray)
    
    #4.3.3 Reserve requirements
    @add_con(core, c36, p_nt_rgu_req[n.n_p, n.t]/n.σ_rgu for n in sc_data.preservearray)
    @add_con!(core, c36, cs.n + L_N_p*(cs.t-1) => -p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c37, p_nt_rgd_req[n.n_p, n.t]/n.σ_rgd for n in sc_data.preservearray)
    @add_con!(core, c37, cs.n + L_N_p*(cs.t-1) => -p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    #assuming c_scr and c_nsc are always positive
    @add_con(core, cmax38, p_jt_pr_max[pr.n_p, pr.t] - p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr; ucon = fill(Inf, size(sc_data.prarray)))
    @add_con(core, c38, p_nt_scr_req[n.n_p, n.t] - n.σ_scr*p_jt_pr_max[n.n_p, n.t] for n in sc_data.preservearray)
    @add_con(core, c39, p_nt_nsc_req[n.n_p, n.t] - n.σ_nsc*p_jt_pr_max[n.n_p, n.t] for n in sc_data.preservearray)

    #4.3.4 Reserve balance
    #Reminder, p and q sets have been split up for pr and cs
    @add_con(core, c40, p_nt_rgu_plus[n.n_p, n.t] - p_nt_rgu_req[n.n_p, n.t] for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c40, pr.n_p + L_N_p*(pr.t-1) => p_jt_rgu_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c40, cs.n_p + L_N_p*(cs.t-1) => p_jt_rgu_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c41, p_nt_rgd_plus[n.n_p, n.t] - p_nt_rgd_req[n.n_p, n.t] for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c41, pr.n_p + L_N_p*(pr.t-1) => p_jt_rgd_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c41, cs.n_p + L_N_p*(cs.t-1) => p_jt_rgd_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c42, p_nt_scr_plus[n.n_p, n.t] - p_nt_rgu_req[n.n_p, n.t] - p_nt_scr_req[n.n_p, n.t] for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c42, pr.n_p + L_N_p*(pr.t-1) => p_jt_rgu_pr[pr.j_pr, pr.t] + p_jt_scr_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c42, cs.n_p + L_N_p*(cs.t-1) => p_jt_rgu_cs[cs.j_cs, cs.t] + p_jt_scr_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c43, p_nt_nsc_plus[n.n_p, n.t] - p_nt_rgu_req[n.n_p, n.t] - p_nt_scr_req[n.n_p, n.t] - p_nt_nsc_req[n.n_p, n.t] for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c43, pr.n_p + L_N_p*(pr.t-1) => p_jt_rgu_pr[pr.j_pr, pr.t] + p_jt_scr_pr[pr.j_pr, pr.t] + p_jt_nsc_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c43, cs.n_p + L_N_p*(cs.t-1) => p_jt_rgu_cs[cs.j_cs, cs.t] + p_jt_scr_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c44, p_nt_rru_plus[n.n_p, n.t] - n.p_rru_min for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c44, pr.n_p + L_N_p*(pr.t-1) => p_jt_rru_on_pr[pr.j_pr, pr.t] + p_jt_rru_off_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c44, cs.n_p + L_N_p*(cs.t-1) => p_jt_rru_on_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c45, p_nt_rrd_plus[n.n_p, n.t] - n.p_rrd_min for n in sc_data.preservearray; ucon = fill(Inf, size(sc_data.preservearray)))
    @add_con!(core, c45, pr.n_p + L_N_p*(pr.t-1) => p_jt_rrd_on_pr[pr.j_pr, pr.t] for pr in sc_data.preservesetarray_pr)
    @add_con!(core, c45, cs.n_p + L_N_p*(cs.t-1) => p_jt_rrd_on_cs[cs.j_cs, cs.t] + p_jt_rrd_off_cs[cs.j_cs, cs.t] for cs in sc_data.preservesetarray_cs)
    @add_con(core, c46, q_nt_qru_plus[n.n_q, n.t] - n.q_qru_min for n in sc_data.qreservearray; ucon = fill(Inf, size(sc_data.qreservearray)))
    @add_con!(core, c46, pr.n_q + L_N_q*(pr.t-1) => q_jt_qru_pr[pr.j_pr, pr.t] for pr in sc_data.qreservesetarray_pr)
    @add_con!(core, c46, cs.n_q + L_N_q*(cs.t-1) => q_jt_qru_cs[cs.j_cs, cs.t] for cs in sc_data.qreservesetarray_cs)
    @add_con(core, c47, q_nt_qrd_plus[n.n_q, n.t] - n.q_qrd_min for n in sc_data.qreservearray; ucon = fill(Inf, size(sc_data.qreservearray)))
    @add_con!(core, c47, pr.n_q + L_N_q*(pr.t-1) => q_jt_qrd_pr[pr.j_pr, pr.t] for pr in sc_data.qreservesetarray_pr)
    @add_con!(core, c47, cs.n_q + L_N_q*(cs.t-1) => q_jt_qrd_cs[cs.j_cs, cs.t] for cs in sc_data.qreservesetarray_cs)

    #skipping constraints 48-67. Assume unit commitment is always satisfied
    
    #4.6.1 Producing and consuming device startup, shutdown, and dispatchable power
    #p_jt variants are split on pr and cs
    @add_con(core, c68_pr, p_jt_on_pr[pr.j_pr, pr.t] + p_jt_su_pr[pr.j_pr, pr.t] + p_jt_sd_pr[pr.j_pr, pr.t] - p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c68_cs, p_jt_on_cs[cs.j_cs, cs.t] + p_jt_su_cs[cs.j_cs, cs.t] + p_jt_sd_cs[cs.j_cs, cs.t] - p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con(core, c69_pr, p_jt_su_pr[pr.j_pr, pr.t] - pr.sum_T_supc_pr_jt for pr in sc_data.prarray)
    @add_con(core, c69_cs, p_jt_su_cs[cs.j_cs, cs.t] - cs.sum_T_supc_cs_jt for cs in sc_data.csarray)
    @add_con(core, c70_pr, p_jt_sd_pr[pr.j_pr, pr.t] - pr.sum_T_sdpc_pr_jt for pr in sc_data.prarray)
    @add_con(core, c70_cs, p_jt_sd_cs[cs.j_cs, cs.t] - cs.sum_T_sdpc_cs_jt for cs in sc_data.csarray)

    #4.6.2 Ramping limits
    #p split for pr and cs
    @add_con(core, c71_pr, p_jt_pr[pr.j_pr, pr.t] - pr.p_0 - pr.dt*(pr.p_ru*(pr.u_on-pr.u_su) + pr.p_ru_su*(pr.u_su+1-pr.u_on)) for pr in sc_data.prarray[1:L_J_pr];
    lcon = fill(-Inf, size(sc_data.prarray[1:L_J_pr])))
    @add_con(core, c71_cs, p_jt_cs[cs.j_cs, cs.t] - cs.p_0 - cs.dt*(cs.p_ru*(cs.u_on-cs.u_su) + cs.p_ru_su*(cs.u_su+1-cs.u_on)) for cs in sc_data.csarray[1:L_J_cs];
    lcon  = fill(-Inf, size(sc_data.csarray[1:L_J_cs])))
    @add_con(core, c72_pr, p_jt_pr[pr.j_pr, pr.t] - p_jt_pr[pr.j_pr, pr.t-1] - pr.dt*(pr.p_ru*(pr.u_on-pr.u_su) + pr.p_ru_su*(pr.u_su+1-pr.u_on)) for pr in sc_data.prarray[L_J_pr+1:end];
    lcon = fill(-Inf, size(sc_data.prarray[L_J_pr+1:end])))
    @add_con(core, c72_cs, p_jt_cs[cs.j_cs, cs.t] - p_jt_cs[cs.j_cs, cs.t-1] - cs.dt*(cs.p_ru*(cs.u_on-cs.u_su) + cs.p_ru_su*(cs.u_su+1-cs.u_on)) for cs in sc_data.csarray[L_J_cs+1:end];
    lcon = fill(-Inf, size(sc_data.csarray[L_J_cs+1:end])))
    @add_con(core, c73_pr, p_jt_pr[pr.j_pr, pr.t] - pr.p_0 + pr.dt*(pr.p_rd*pr.u_on+pr.p_rd_sd*(1-pr.u_on)) for pr in sc_data.prarray[1:L_J_pr];
    ucon = fill(Inf, size(sc_data.prarray[1:L_J_pr])))
    @add_con(core, c73_cs, p_jt_cs[cs.j_cs, cs.t] - cs.p_0 + cs.dt*(cs.p_rd*cs.u_on+cs.p_rd_sd*(1-cs.u_on)) for cs in sc_data.csarray[1:L_J_cs];
    ucon = fill(Inf, size(sc_data.csarray[1:L_J_cs])))
    @add_con(core, c74_pr, p_jt_pr[pr.j_pr, pr.t] - p_jt_pr[pr.j_pr, pr.t-1] + pr.dt*(pr.p_rd*pr.u_on+pr.p_rd_sd*(1-pr.u_on)) for pr in sc_data.prarray[L_J_pr+1:end];
    ucon = fill(Inf, size(sc_data.prarray[L_J_pr+1:end])))
    @add_con(core, c74_cs, p_jt_cs[cs.j_cs, cs.t] - p_jt_cs[cs.j_cs, cs.t-1] + cs.dt*(cs.p_rd*cs.u_on+cs.p_rd_sd*(1-cs.u_on)) for cs in sc_data.csarray[L_J_cs+1:end];
    ucon = fill(Inf, size(sc_data.csarray[L_J_cs+1:end])))

    #4.6.3 Maximum/minimum energy over multiple intervals
    #J_pr,cs has been split for pr and cs
    @add_con(core, c75_pr, e_w_plus_max_pr[w.w_en_max_pr_ind] + w.e_max for w in sc_data.W_en_max_pr; ucon = fill(Inf, size(sc_data.W_en_max_pr)))
    @add_con!(core, c75_pr, t.w_en_max_pr_ind => -t.dt*p_jt_pr[t.j_pr, t.t] for t in sc_data.T_w_en_max_pr)
    @add_con(core, c75_cs, e_w_plus_max_cs[w.w_en_max_cs_ind] + w.e_max for w in sc_data.W_en_max_cs; ucon = fill(Inf, size(sc_data.W_en_max_cs)))
    @add_con!(core, c75_cs, t.w_en_max_cs_ind => -t.dt*p_jt_cs[t.j_cs, t.t] for t in sc_data.T_w_en_max_cs)
    @add_con(core, c76_pr, e_w_plus_min_pr[w.w_en_min_pr_ind] - w.e_min for w in sc_data.W_en_min_pr; lcon = fill(-Inf, size(sc_data.W_en_min_pr)))
    @add_con!(core, c76_pr, t.w_en_min_pr_ind => -t.dt*p_jt_pr[t.j_pr, t.t] for t in sc_data.T_w_en_min_pr)
    @add_con(core, c76_cs, e_w_plus_min_cs[w.w_en_min_cs_ind] - w.e_min for w in sc_data.W_en_min_cs; lcon = fill(-Inf, size(sc_data.W_en_min_cs)))
    @add_con!(core, c76_cs, t.w_en_min_cs_ind => -t.dt*p_jt_cs[t.j_cs, t.t] for t in sc_data.T_w_en_min_cs)
    @add_con(core, c78_pr, z_w_en_max_pr[w.w_en_max_pr_ind] - sum(sc_data.c_e)*e_w_plus_max_pr[w.w_en_max_pr_ind] for w in sc_data.W_en_max_pr)
    @add_con(core, c78_cs, z_w_en_max_cs[w.w_en_max_cs_ind] - sum(sc_data.c_e)*e_w_plus_max_cs[w.w_en_max_cs_ind] for w in sc_data.W_en_max_cs)
    @add_con(core, c79_pr, z_w_en_min_pr[w.w_en_min_pr_ind] - sum(sc_data.c_e)*e_w_plus_min_pr[w.w_en_min_pr_ind] for w in sc_data.W_en_min_pr)
    @add_con(core, c79_cs, z_w_en_min_cs[w.w_en_min_cs_ind] - sum(sc_data.c_e)*e_w_plus_min_cs[w.w_en_min_cs_ind] for w in sc_data.W_en_min_cs)

    #4.6.5 Device reserve costs
    #p_jt split into pr and cs
    @add_con(core, c90_pr, z_jt_rgu_pr[pr.j_pr, pr.t] - pr.dt*pr.c_rgu*p_jt_rgu_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c90_cs, z_jt_rgu_cs[cs.j_cs, cs.t] - cs.dt*cs.c_rgu*p_jt_rgu_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con(core, c91_pr, z_jt_rgd_pr[pr.j_pr, pr.t] - pr.dt*pr.c_rgd*p_jt_rgd_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c91_cs, z_jt_rgd_cs[cs.j_cs, cs.t] - cs.dt*cs.c_rgd*p_jt_rgd_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con(core, c92_pr, z_jt_scr_pr[pr.j_pr, pr.t] - pr.dt*pr.c_scr*p_jt_scr_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c92_cs, z_jt_scr_cs[cs.j_cs, cs.t] - cs.dt*cs.c_scr*p_jt_scr_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con(core, c93_pr, z_jt_nsc_pr[pr.j_pr, pr.t] - pr.dt*pr.c_nsc*p_jt_nsc_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    #due to c106, z_jt_nsc_cs = 0. This is realized in o6
    @add_con(core, c94_pr, z_jt_rru_pr[pr.j_pr, pr.t] - pr.dt*(pr.c_rru_on*p_jt_rru_on_pr[pr.j_pr, pr.t] + pr.c_rru_off*p_jt_rru_off_pr[pr.j_pr, pr.t]) for pr in sc_data.prarray)
    @add_con(core, c94_cs, z_jt_rru_cs[cs.j_cs, cs.t] - cs.dt*(cs.c_rru_on*p_jt_rru_on_cs[cs.j_cs, cs.t]) for cs in sc_data.csarray)
    @add_con(core, c95_pr, z_jt_rrd_pr[pr.j_pr, pr.t] - pr.dt*(pr.c_rrd_on*p_jt_rrd_on_pr[pr.j_pr, pr.t]) for pr in sc_data.prarray)
    @add_con(core, c95_cs, z_jt_rrd_cs[cs.j_cs, cs.t] - cs.dt*(cs.c_rrd_on*p_jt_rrd_on_cs[cs.j_cs, cs.t] + cs.c_rrd_off*p_jt_rrd_off_cs[cs.j_cs, cs.t]) for cs in sc_data.csarray)
    @add_con(core, c96_pr, z_jt_qru_pr[pr.j_pr, pr.t] - pr.dt*pr.c_qru*q_jt_qru_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c96_cs, z_jt_qru_cs[cs.j_cs, cs.t] - cs.dt*cs.c_qru*q_jt_qru_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con(core, c97_pr, z_jt_qrd_pr[pr.j_pr, pr.t] - pr.dt*pr.c_qrd*q_jt_qrd_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con(core, c97_cs, z_jt_qrd_cs[cs.j_cs, cs.t] - cs.dt*cs.c_qrd*q_jt_qrd_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)

    #4.6.6 Absolute reserve limits, based on ramp rates
    @add_con(core, c98_pr, p_jt_rgu_pr[pr.j_pr, pr.t] - pr.p_rgu_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c98_cs, p_jt_rgu_cs[cs.j_cs, cs.t] - cs.p_rgu_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c99_pr, p_jt_rgd_pr[pr.j_pr, pr.t] - pr.p_rgd_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c99_cs, p_jt_rgd_cs[cs.j_cs, cs.t] - cs.p_rgd_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c100_pr, p_jt_rgu_pr[pr.j_pr, pr.t] + p_jt_scr_pr[pr.j_pr, pr.t] - pr.p_scr_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c100_cs, p_jt_rgu_cs[cs.j_cs, cs.t] + p_jt_scr_cs[cs.j_cs, cs.t] - cs.p_scr_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c101_pr, p_jt_nsc_pr[pr.j_pr, pr.t] - pr.p_nsc_max*(1-pr.u_on) for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    #due to c107, c101_cs is not needed (assume unit commitment satisfies it)
    @add_con(core, c102_pr, p_jt_rgu_pr[pr.j_pr, pr.t] + p_jt_scr_pr[pr.j_pr, pr.t] + p_jt_rru_on_pr[pr.j_pr, pr.t] - pr.p_rru_on_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c102_cs, p_jt_rgu_cs[cs.j_cs, cs.t] + p_jt_scr_cs[cs.j_cs, cs.t] + p_jt_rru_on_cs[cs.j_cs, cs.t] - cs.p_rru_on_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c103_pr, p_jt_nsc_pr[pr.j_pr, pr.t] + p_jt_rru_off_pr[pr.j_pr, pr.t] - pr.p_rru_off_max*(1-pr.u_on) for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    #due to c107, 108, c103_cs is not needed (assume unit commitment satisfies it)
    @add_con(core, c104_pr, p_jt_rgd_pr[pr.j_pr, pr.t] + p_jt_rrd_on_pr[pr.j_pr, pr.t] - pr.p_rrd_on_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c104_cs, p_jt_rgd_cs[cs.j_cs, cs.t] + p_jt_rrd_on_cs[cs.j_cs, cs.t] - cs.p_rrd_on_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    #due to c106, c105_pr is not needed (assume unit commitment satisfies it)
    @add_con(core, c105_cs, p_jt_rrd_off_cs[cs.j_cs, cs.t] - cs.p_rrd_off_max*(1-cs.u_on) for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    #c106 forces p_jt_rrd_off_pr = 0. This is realized in changes to c45, c95, and c105
    #c107 forces p_jt_nsc_cs = 0. This is realized in changes to c43, c93, c101, and c103
    #c108 forces p_jt_rru_off_cs = 0. This is realized in changes to c44, c94, and c103

    #4.6.7 Relative reserve limits, based on headroom to max/min, producing devices
    @add_con(core, c109, p_jt_on_pr[pr.j_pr, pr.t] + p_jt_rgu_pr[pr.j_pr, pr.t] + p_jt_scr_pr[pr.j_pr, pr.t] + p_jt_rru_on_pr[pr.j_pr, pr.t] - pr.p_max*pr.u_on for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c110, p_jt_on_pr[pr.j_pr, pr.t] - p_jt_rgd_pr[pr.j_pr, pr.t] - p_jt_rrd_on_pr[pr.j_pr, pr.t] - pr.p_min*pr.u_on for pr in sc_data.prarray; ucon = fill(Inf, size(sc_data.prarray)))
    @add_con(core, c111, p_jt_su_pr[pr.j_pr, pr.t] + p_jt_sd_pr[pr.j_pr, pr.t] + p_jt_nsc_pr[pr.j_pr, pr.t] + p_jt_rru_off_pr[pr.j_pr, pr.t] - pr.p_max*(1-pr.u_on) for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c112, q_jt_pr[pr.j_pr, pr.t] + q_jt_qru_pr[pr.j_pr, pr.t] - pr.q_max*(pr.u_on + pr.sum2_T_supc_pr_jt + pr.sum2_T_sdpc_pr_jt) for pr in sc_data.prarray; lcon = fill(-Inf, size(sc_data.prarray)))
    @add_con(core, c113, q_jt_pr[pr.j_pr, pr.t] - q_jt_qrd_pr[pr.j_pr, pr.t] - pr.q_min*(pr.u_on + pr.sum2_T_supc_pr_jt + pr.sum2_T_sdpc_pr_jt) for pr in sc_data.prarray; ucon = fill(Inf, size(sc_data.prarray)))

    @add_con(core, c114, q_jt_pr[pr.j_pr, pr.t] + q_jt_qru_pr[pr.j_pr, pr.t] - pr.q_max_p0*(pr.u_on + pr.sum2_T_supc_pr_jt + pr.sum2_T_sdpc_pr_jt) - pr.beta_max*p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray_pqbounds;
    lcon = fill(-Inf, size(sc_data.prarray_pqbounds)))
    @add_con(core, c115, q_jt_pr[pr.j_pr, pr.t] - q_jt_qrd_pr[pr.j_pr, pr.t] - pr.q_min_p0*(pr.u_on + pr.sum2_T_supc_pr_jt + pr.sum2_T_sdpc_pr_jt) - pr.beta_min*p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray_pqbounds;
    ucon = fill(Inf, size(sc_data.prarray_pqbounds)))
    @add_con(core, c116, q_jt_pr[pr.j_pr, pr.t] - pr.q_p0*(pr.u_on + pr.sum2_T_supc_pr_jt + pr.sum2_T_sdpc_pr_jt) - pr.beta*p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray_pqe)
    #These constraints could be removed and the variables removed to simplify other constraints. However, they are kept for continuity
    @add_con(core, c117, q_jt_qru_pr[pr.j_pr, pr.t] for pr in sc_data.prarray_pqe)
    @add_con(core, c118, q_jt_qrd_pr[pr.j_pr, pr.t] for pr in sc_data.prarray_pqe)

    #4.6.8 Relative reserve limits, based on headroom to max/min, consuming devices
    @add_con(core, c119, p_jt_on_cs[cs.j_cs, cs.t] + p_jt_rgd_cs[cs.j_cs, cs.t] + p_jt_rrd_on_cs[cs.j_cs, cs.t] - cs.p_max*cs.u_on for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c120, p_jt_on_cs[cs.j_cs, cs.t] - p_jt_rgu_cs[cs.j_cs, cs.t] - p_jt_scr_cs[cs.j_cs, cs.t] - p_jt_rru_on_cs[cs.j_cs, cs.t] -cs.p_min*cs.u_on for cs in sc_data.csarray; ucon = fill(Inf, size(sc_data.csarray)))
    @add_con(core, c121, p_jt_su_cs[cs.j_cs, cs.t] + p_jt_sd_cs[cs.j_cs, cs.t] + p_jt_rrd_off_cs[cs.j_cs, cs.t] - cs.p_max*(1-cs.u_on) for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c122, q_jt_cs[cs.j_cs, cs.t] + q_jt_qrd_cs[cs.j_cs, cs.t] - cs.q_max*(cs.u_on + cs.sum2_T_supc_cs_jt + cs.sum2_T_sdpc_cs_jt) for cs in sc_data.csarray; lcon = fill(-Inf, size(sc_data.csarray)))
    @add_con(core, c123, q_jt_cs[cs.j_cs, cs.t] - q_jt_qru_cs[cs.j_cs, cs.t] - cs.q_min*(cs.u_on + cs.sum2_T_supc_cs_jt + cs.sum2_T_sdpc_cs_jt) for cs in sc_data.csarray; ucon = fill(Inf, size(sc_data.csarray)))
    @add_con(core, c124, q_jt_cs[cs.j_cs, cs.t] + q_jt_qrd_cs[cs.j_cs, cs.t] - cs.q_max_p0*(cs.u_on + cs.sum2_T_supc_cs_jt + cs.sum2_T_sdpc_cs_jt) - cs.beta_max*p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray_pqbounds; lcon = fill(-Inf, size(sc_data.csarray_pqbounds)))
    @add_con(core, c125, q_jt_cs[cs.j_cs, cs.t] - q_jt_qru_cs[cs.j_cs, cs.t] - cs.q_min_p0*(cs.u_on + cs.sum2_T_supc_cs_jt + cs.sum2_T_sdpc_cs_jt) - cs.beta_min*p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray_pqbounds; ucon = fill(Inf, size(sc_data.csarray_pqbounds)))
    @add_con(core, c126, q_jt_cs[cs.j_cs, cs.t] - cs.q_p0*(cs.u_on + cs.sum2_T_supc_cs_jt + cs.sum2_T_sdpc_cs_jt) - cs.beta*p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray_pqe)
    @add_con(core, c127, q_jt_qru_cs[cs.j_cs, cs.t] for cs in sc_data.csarray_pqe)
    @add_con(core, c128, q_jt_qrd_cs[cs.j_cs, cs.t] for cs in sc_data.csarray_pqe)

    #4.6.9 Energy cost and value
    @add_con(core, c130_pr, p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
    @add_con!(core, c130_pr, pr.j_pr + L_J_pr*(pr.t-1) => -p_jtm_pr[pr.flat_k] for pr in sc_data.p_jtm_flattened_pr)
    @add_con(core, c130_cs, p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
    @add_con!(core, c130_cs, cs.j_cs + L_J_cs*(cs.t-1) => -p_jtm_cs[cs.flat_k] for cs in sc_data.p_jtm_flattened_cs)
    @add_con(core, c131_pr, z_jt_en_pr[pr.j_pr, pr.t]/pr.dt for pr in sc_data.prarray)
    @add_con!(core, c131_pr, pr.j_pr + L_J_pr*(pr.t-1) => -pr.c_en*p_jtm_pr[pr.flat_k] for pr in sc_data.p_jtm_flattened_pr)
    @add_con(core, c131_cs, z_jt_en_cs[cs.j_cs, cs.t]/cs.dt for cs in sc_data.csarray)
    @add_con!(core, c131_cs, cs.j_cs + L_J_cs*(cs.t-1) => -cs.c_en*p_jtm_cs[cs.flat_k] for cs in sc_data.p_jtm_flattened_cs)

    #4.7 Shunt devices
    @add_con(core, c132, p_jt_sh[sh.j_sh, sh.t] - g_jt_sh[sh.j_sh, sh.t]*v_it[sh.bus, sh.t]^2 for sh in sc_data.shuntarray)
    @add_con(core, c133, q_jt_sh[sh.j_sh, sh.t] + b_jt_sh[sh.j_sh, sh.t]*v_it[sh.bus, sh.t]^2 for sh in sc_data.shuntarray)
    @add_con(core, c134, g_jt_sh[sh.j_sh, sh.t] - sh.g_sh*sh.u_sh for sh in sc_data.shuntarray)
    @add_con(core, c135, b_jt_sh[sh.j_sh, sh.t] - sh.b_sh*sh.u_sh for sh in sc_data.shuntarray)
    #Assume (136-137) properly handled in uc solution

    #4.8.1 AC branch flow limits and penalties
    #AC branches split into ln and xf
    @add_con(core, c139_ln, z_jt_s_ln[ln.j_ln, ln.t] - ln.dt*sum(sc_data.c_s)*s_jt_plus_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray)
    @add_con(core, c139_xf, z_jt_s_xf[xf.j_xf, xf.t] -xf.dt*sum(sc_data.c_s)*s_jt_plus_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray)
    @add_con(core, c140_ln, (p_jt_fr_ln[ln.j_ln, ln.t]^2 + q_jt_fr_ln[ln.j_ln, ln.t]^2)^.5 - ln.s_max - s_jt_plus_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray;
                        lcon = fill(-Inf, size(sc_data.aclbrancharray)))
    @add_con(core, c140_xf, (p_jt_fr_xf[xf.j_xf, xf.t]^2 + q_jt_fr_xf[xf.j_xf, xf.t]^2)^.5 - xf.s_max - s_jt_plus_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray;
                        lcon = fill(-Inf, size(sc_data.acxbrancharray)))
    @add_con(core, c141_ln, (p_jt_to_ln[ln.j_ln, ln.t]^2 + q_jt_to_ln[ln.j_ln, ln.t]^2)^.5 - ln.s_max - s_jt_plus_ln[ln.j_ln, ln.t] for ln in sc_data.aclbrancharray;
                        lcon = fill(-Inf, size(sc_data.aclbrancharray)))
    @add_con(core, c141_xf, (p_jt_to_xf[xf.j_xf, xf.t]^2 + q_jt_to_xf[xf.j_xf, xf.t]^2)^.5 - xf.s_max - s_jt_plus_xf[xf.j_xf, xf.t] for xf in sc_data.acxbrancharray;
                        lcon = fill(-Inf, size(sc_data.acxbrancharray)))

    #4.8.2 AC branch controls
    #c142 forces φ_jt_ln = 0. This is realized in c148-151 and c161
    #c143 forces τ_jt_ln = 1. This is realized in c148-151
    @add_con(core, c144, φ_jt_xf[xf.j_xf, xf.t] - xf.phi_o for xf in sc_data.fpdarray)
    @add_con(core, c145, τ_jt_xf[xf.j_xf, xf.t] - xf.tau_o for xf in sc_data.fwrarray)
    @add_con(core, c146, φ_jt_xf[xf.j_xf, xf.t] for xf in sc_data.vpdarray;
                lcon = [xf.phi_min for xf in sc_data.vpdarray], ucon = [xf.phi_max for xf in sc_data.vpdarray])
    @add_con(core, c147, τ_jt_xf[xf.j_xf, xf.t] for xf in sc_data.vwrarray;
                lcon = [xf.tau_min for xf in sc_data.vwrarray], ucon = [xf.tau_max for xf in sc_data.vwrarray])

    #4.8.3 AC branch flows
    
    @add_con(core, c148_ln, -p_jt_fr_ln[ln.j_ln, ln.t] + ln.u_on*((ln.g_sr+ln.g_fr)*v_it[ln.fr_bus, ln.t]^2 + (-ln.g_sr*cos(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t])
     - ln.b_sr*sin(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]))*v_it[ln.fr_bus, ln.t]*v_it[ln.to_bus, ln.t]) for ln in sc_data.aclbrancharray)
    @add_con(core, c148_xf, -p_jt_fr_xf[xf.j_xf, xf.t] + xf.u_on*((xf.g_sr+xf.g_fr)*v_it[xf.fr_bus, xf.t]^2/(τ_jt_xf[xf.j_xf, xf.t]^2) + (-xf.g_sr*cos(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t])
     - xf.b_sr*sin(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]))*v_it[xf.fr_bus, xf.t]*v_it[xf.to_bus, xf.t]/τ_jt_xf[xf.j_xf, xf.t]) for xf in sc_data.acxbrancharray)
    @add_con(core, c149_ln, -q_jt_fr_ln[ln.j_ln, ln.t] + ln.u_on*((-ln.b_sr - ln.b_fr - ln.b_ch/2)*v_it[ln.fr_bus, ln.t]^2 + (ln.b_sr*cos(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]) 
    - ln.g_sr*sin(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]))*v_it[ln.fr_bus, ln.t]*v_it[ln.to_bus, ln.t]) for ln in sc_data.aclbrancharray)
    @add_con(core, c149_xf, -q_jt_fr_xf[xf.j_xf, xf.t] + xf.u_on*((-xf.b_sr - xf.b_fr - xf.b_ch/2)*v_it[xf.fr_bus, xf.t]^2/(τ_jt_xf[xf.j_xf, xf.t]^2) + (xf.b_sr*cos(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]) 
    - xf.g_sr*sin(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]))*v_it[xf.fr_bus, xf.t]*v_it[xf.to_bus, xf.t]/τ_jt_xf[xf.j_xf, xf.t]) for xf in sc_data.acxbrancharray)
    @add_con(core, c150_ln, -p_jt_to_ln[ln.j_ln, ln.t] + ln.u_on*((ln.g_sr+ln.g_to)*v_it[ln.to_bus, ln.t]^2 + (-ln.g_sr*cos(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]) 
    + ln.b_sr*sin(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]))*v_it[ln.fr_bus, ln.t]*v_it[ln.to_bus, ln.t]) for ln in sc_data.aclbrancharray)
    @add_con(core, c150_xf, -p_jt_to_xf[xf.j_xf, xf.t] + xf.u_on*((xf.g_sr+xf.g_to)*v_it[xf.to_bus, xf.t]^2 + (-xf.g_sr*cos(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]) 
    + xf.b_sr*sin(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]))*v_it[xf.fr_bus, xf.t]*v_it[xf.to_bus, xf.t]/τ_jt_xf[xf.j_xf, xf.t]) for xf in sc_data.acxbrancharray)
    @add_con(core, c151_ln, -q_jt_to_ln[ln.j_ln, ln.t] + ln.u_on*((-ln.b_sr-ln.b_to-ln.b_ch/2)*v_it[ln.to_bus, ln.t]^2 + (ln.b_sr*cos(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]) 
    + ln.g_sr*sin(θ_it[ln.fr_bus, ln.t] - θ_it[ln.to_bus, ln.t]))*v_it[ln.fr_bus, ln.t]*v_it[ln.to_bus, ln.t]) for ln in sc_data.aclbrancharray)
    @add_con(core, c151_xf, -q_jt_to_xf[xf.j_xf, xf.t] + xf.u_on*((-xf.b_sr-xf.b_to-xf.b_ch/2)*v_it[xf.to_bus, xf.t]^2 + (xf.b_sr*cos(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]) 
    + xf.g_sr*sin(θ_it[xf.fr_bus, xf.t] - θ_it[xf.to_bus, xf.t] - φ_jt_xf[xf.j_xf, xf.t]))*v_it[xf.fr_bus, xf.t]*v_it[xf.to_bus, xf.t]/τ_jt_xf[xf.j_xf, xf.t]) for xf in sc_data.acxbrancharray)

    #4.8.4 DC lines
    @add_con(core, c156, p_jt_fr_dc[dc.j_dc, dc.t] + p_jt_to_dc[dc.j_dc, dc.t] for dc in sc_data.dclinearray)

    #4.9.1 Penalty on post-contingency AC branch overload
    if include_ctg
        #ac split into ln and xf
        @add_con(core, c158_ln, ln.dt*sum(sc_data.c_s)*s_jtk_plus_ln[ln.flat_jtk_ln] - z_jtk_s_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened)
        @add_con(core, c158_xf, xf.dt*sum(sc_data.c_s)*s_jtk_plus_xf[xf.flat_jtk_xf] - z_jtk_s_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened)

        #4.9.2 Post-contingency AC power flow limits
        @add_con(core, c159_ln, (p_jtk_ln[ln.flat_jtk_ln]^2 + q_jt_fr_ln[ln.j_ln, ln.t]^2)^0.5 - ln.s_max_ctg - s_jtk_plus_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened;
        lcon = fill(-Inf, size(sc_data.jtk_ln_flattened)))
        @add_con(core, c159_xf, (p_jtk_xf[xf.flat_jtk_xf]^2 + q_jt_fr_xf[xf.j_xf, xf.t]^2)^0.5 - xf.s_max_ctg - s_jtk_plus_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened;
        lcon = fill(-Inf, size(sc_data.jtk_xf_flattened)))
        @add_con(core, c160_ln, (p_jtk_ln[ln.flat_jtk_ln]^2 + q_jt_to_ln[ln.j_ln, ln.t]^2)^0.5 - ln.s_max_ctg - s_jtk_plus_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened;
        lcon = fill(-Inf, size(sc_data.jtk_ln_flattened)))
        @add_con(core, c160_xf, (p_jtk_xf[xf.flat_jtk_xf]^2 + q_jt_to_xf[xf.j_xf, xf.t]^2)^0.5 - xf.s_max_ctg - s_jtk_plus_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened;
        lcon = fill(-Inf, size(sc_data.jtk_xf_flattened)))

        #4.9.3 Post-contingency AC branch real power flows
        @add_con(core, c161_ln, p_jtk_ln[ln.flat_jtk_ln] + ln.b_sr*ln.u_on*(θ_itk[ln.fr_bus, ln.t, ln.ctg] - θ_itk[ln.to_bus, ln.t, ln.ctg]) for ln in sc_data.jtk_ln_flattened)
        @add_con(core, c161_xf, p_jtk_xf[xf.flat_jtk_xf] + xf.b_sr*xf.u_on*(θ_itk[xf.fr_bus, xf.t, xf.ctg] - θ_itk[xf.to_bus, xf.t, xf.ctg] - φ_jt_xf[xf.j_xf, xf.t]) for xf in sc_data.jtk_xf_flattened)

        #4.9.4 Post-contingency real power balance
        
        @add_con(core, c162, p_t_sl[t] for t in 1:L_T)
        @add_con!(core, c162, pr.t => -p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.prarray)
        @add_con!(core, c162, cs.t => p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.csarray)
        @add_con!(core, c162, sh.t => p_jt_sh[sh.j_sh, sh.t] for sh in sc_data.shuntarray)
        @add_con(core, c163, -p_t_sl[b.t]/I for b in sc_data.k_busarray) #alpha_i = 1/I as per the data format guidelines
        @add_con!(core, c163, ln.fr_bus + I*(ln.t-1) + I*L_T*(ln.ctg-1) => -p_jtk_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened)
        @add_con!(core, c163, ln.to_bus + I*(ln.t-1) + I*L_T*(ln.ctg-1) => p_jtk_ln[ln.flat_jtk_ln] for ln in sc_data.jtk_ln_flattened)
        @add_con!(core, c163, xf.fr_bus + I*(xf.t-1) + I*L_T*(xf.ctg-1) => -p_jtk_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened)
        @add_con!(core, c163, xf.to_bus + I*(xf.t-1) + I*L_T*(xf.ctg-1) => p_jtk_xf[xf.flat_jtk_xf] for xf in sc_data.jtk_xf_flattened)
        @add_con!(core, c163, pr.bus + I*(pr.t-1) + I*L_T*(pr.k-1) => p_jt_pr[pr.j_pr, pr.t] for pr in sc_data.k_prarray)
        @add_con!(core, c163, cs.bus + I*(cs.t-1) + I*L_T*(cs.k-1) => -p_jt_cs[cs.j_cs, cs.t] for cs in sc_data.k_csarray)
        @add_con!(core, c163, sh.bus + I*(sh.t-1) + I*L_T*(sh.k-1) => -p_jt_sh[sh.j_sh, sh.t] for sh in sc_data.k_shuntarray)
        @add_con!(core, c163, dc.fr_bus + I*(dc.t-1) + I*L_T*(dc.ctg-1) => -p_jt_fr_dc[dc.j_dc, dc.t] for dc in sc_data.jtk_dc_flattened)
        @add_con!(core, c163, dc.to_bus + I*(dc.t-1) + I*L_T*(dc.ctg-1) => p_jt_to_dc[dc.j_dc, dc.t] for dc in sc_data.jtk_dc_flattened)
        
    end


    vars = (b_jt_sh = b_jt_sh,
            g_jt_sh = g_jt_sh,
            #Split e_w_plus into separate sets for W_en_min and W_en_max ad for pr, cs
            e_w_plus_min_pr = e_w_plus_min_pr,
            e_w_plus_min_cs = e_w_plus_min_cs,
            e_w_plus_max_pr = e_w_plus_max_pr,
            e_w_plus_max_cs = e_w_plus_max_cs,
            p_it = p_it,
            p_it_plus = p_it_plus,
            #splitting p_jt and q_jt for shunts, producers, and consumers
            p_jt_sh = p_jt_sh,
            p_jt_pr = p_jt_pr,
            p_jt_cs = p_jt_cs,
            q_jt_sh = q_jt_sh,
            q_jt_pr = q_jt_pr,
            q_jt_cs = q_jt_cs,
            #Splitting p on, sd , su into pr and cs
            p_jt_on_pr = p_jt_on_pr,
            p_jt_on_cs = p_jt_on_cs,
            p_jt_su_pr = p_jt_su_pr,
            p_jt_su_cs = p_jt_su_cs,
            p_jt_sd_pr = p_jt_sd_pr,
            p_jt_sd_cs = p_jt_sd_cs,
            #p_jtm has been flattened and uses only one special index, k_flat
            p_jtm_pr = p_jtm_pr,
            p_jtm_cs = p_jtm_cs,
            #to/from power split into ln, xf, and dc lines
            p_jt_fr_ln = p_jt_fr_ln,
            p_jt_fr_xf = p_jt_fr_xf,
            p_jt_fr_dc = p_jt_fr_dc,
            p_jt_to_ln = p_jt_to_ln,
            p_jt_to_xf = p_jt_to_xf,
            p_jt_to_dc = p_jt_to_dc,
            q_jt_fr_ln = q_jt_fr_ln,
            q_jt_fr_xf = q_jt_fr_xf,
            q_jt_fr_dc = q_jt_fr_dc,
            q_jt_to_ln = q_jt_to_ln,
            q_jt_to_xf = q_jt_to_xf,
            q_jt_to_dc = q_jt_to_dc,
            #p_jt rgu, rgd, scr, rru,on, rru,off, rrd,on, rrd,off and q_jt qru/qrd split into pr and cs
            p_jt_rgu_pr = p_jt_rgu_pr,
            p_jt_rgu_cs = p_jt_rgu_cs,
            p_jt_rgd_pr = p_jt_rgd_pr,
            p_jt_rgd_cs = p_jt_rgd_cs,
            p_jt_scr_pr = p_jt_scr_pr,
            p_jt_scr_cs = p_jt_scr_cs,
            p_jt_nsc_pr = p_jt_nsc_pr,
            p_jt_rru_on_pr = p_jt_rru_on_pr,
            p_jt_rru_on_cs = p_jt_rru_on_cs,
            p_jt_rru_off_pr = p_jt_rru_off_pr,
            p_jt_rrd_on_pr = p_jt_rrd_on_pr,
            p_jt_rrd_on_cs = p_jt_rrd_on_cs,
            p_jt_rrd_off_cs = p_jt_rrd_off_cs,
            q_jt_qru_pr = q_jt_qru_pr,
            q_jt_qru_cs = q_jt_qru_cs,
            q_jt_qrd_pr = q_jt_qrd_pr,
            q_jt_qrd_cs = q_jt_qrd_cs,
            p_nt_rgu_req = p_nt_rgu_req,
            p_nt_rgd_req = p_nt_rgd_req,
            p_nt_scr_req = p_nt_scr_req,
            p_nt_nsc_req = p_nt_nsc_req,
            p_jt_pr_max = p_jt_pr_max,
            p_nt_rgu_plus = p_nt_rgu_plus,
            p_nt_rgd_plus = p_nt_rgd_plus,
            p_nt_scr_plus = p_nt_scr_plus,
            p_nt_nsc_plus = p_nt_nsc_plus,
            p_nt_rru_plus = p_nt_rru_plus,
            p_nt_rrd_plus = p_nt_rrd_plus,
            q_nt_qru_plus = q_nt_qru_plus,
            q_nt_qrd_plus = q_nt_qrd_plus,
            q_it = q_it,
            q_it_plus = q_it_plus,
            #s_jt_plus split on ln and xf
            s_jt_plus_ln = s_jt_plus_ln,
            s_jt_plus_xf = s_jt_plus_xf,
            v_it = v_it,
            z_w_en_max_pr = z_w_en_max_pr,
            z_w_en_max_cs = z_w_en_max_cs,
            z_w_en_min_pr = z_w_en_min_pr,
            z_w_en_min_cs = z_w_en_min_cs,
            #split z_jt_en and on into pr and cs
            z_jt_en_pr = z_jt_en_pr,
            z_jt_en_cs = z_jt_en_cs,
            z_it_p = z_it_p,
            z_it_q = z_it_q,
            #z_jt_s split into ln and xf
            z_jt_s_ln = z_jt_s_ln,
            z_jt_s_xf = z_jt_s_xf,
            #z_jt rgu, rgd, scr, nsc, rru, rrd, qru, qrd split into pr and cs
            z_jt_rgu_pr = z_jt_rgu_pr,
            z_jt_rgu_cs = z_jt_rgu_cs,
            z_jt_rgd_pr = z_jt_rgd_pr,
            z_jt_rgd_cs = z_jt_rgd_cs,
            z_jt_scr_pr = z_jt_scr_pr,
            z_jt_scr_cs = z_jt_scr_cs,
            z_jt_nsc_pr = z_jt_nsc_pr,
            z_jt_rru_pr = z_jt_rru_pr,
            z_jt_rru_cs = z_jt_rru_cs,
            z_jt_rrd_pr = z_jt_rrd_pr,
            z_jt_rrd_cs = z_jt_rrd_cs,
            z_jt_qru_pr = z_jt_qru_pr,
            z_jt_qru_cs = z_jt_qru_cs,
            z_jt_qrd_pr = z_jt_qrd_pr,
            z_jt_qrd_cs = z_jt_qrd_cs,
            z_nt_rgu = z_nt_rgu,
            z_nt_rgd = z_nt_rgd,
            z_nt_scr = z_nt_scr,
            z_nt_nsc = z_nt_nsc,
            z_nt_rru = z_nt_rru,
            z_nt_rrd = z_nt_rrd,
            z_nt_qru = z_nt_qru,
            z_nt_qrd = z_nt_qrd,
            θ_it = θ_it,
            #split τjt and φjt into xf only, ln is fixed
            τ_jt_xf = τ_jt_xf,
            φ_jt_xf = φ_jt_xf)

        cons = (
            c11 = c11,
            c12 = c12,
            c13 = c13,
            c14 = c14,
            c15 = c15,
            c16 = c16,
            c17 = c17,
            c18 = c18,
            c28 = c28,
            c29 = c29,
            c30 = c30,
            c31 = c31,
            c32 = c32,
            c33 = c33,
            c34 = c34,
            c35 = c35,
            c36 = c36,
            c37 = c37,
            cmax38 = cmax38,
            c38 = c38,
            c39 = c39,
            c40 = c40,
            c41 = c41,
            c42 = c42,
            c43 = c43,
            c44 = c44,
            c45 = c45,
            c46 = c46,
            c47 = c47,
            c68_pr = c68_pr,
            c68_cs = c68_cs,
            c69_pr = c69_pr,
            c69_cs = c69_cs,
            c70_pr = c70_pr,
            c70_cs = c70_cs,
            c71_pr = c71_pr,
            c71_cs = c71_cs,
            c72_pr = c72_pr,
            c72_cs = c72_cs,
            c73_pr = c73_pr,
            c73_cs = c73_cs,
            c74_pr = c74_pr,
            c74_cs = c74_cs,
            c75_pr = c75_pr,
            c75_cs = c75_cs,
            c76_pr = c76_pr,
            c76_cs = c76_cs,
            c78_pr = c78_pr,
            c78_cs = c78_cs,
            c79_pr = c79_pr,
            c79_cs = c79_cs,
            c94_pr = c94_pr,
            c94_cs = c94_cs,
            c95_pr = c95_pr,
            c95_cs = c95_cs,
            c96_pr = c96_pr,
            c96_cs = c96_cs,
            c97_pr = c97_pr,
            c97_cs = c97_cs,
            c98_pr = c98_pr,
            c98_cs = c98_cs,
            c99_pr = c99_pr,
            c99_cs = c99_cs,
            c100_pr = c100_pr,
            c100_cs = c100_cs,
            c101_pr = c101_pr,
            c102_pr = c102_pr,
            c102_cs = c102_cs,
            c103_pr = c103_pr,
            c104_pr = c104_pr,
            c104_cs = c104_cs,
            c105_cs = c105_cs,
            c109 = c109,
            c110 = c110,
            c111 = c111,
            c112 = c112,
            c113 = c113,
            c114 = c114,
            c115 = c115,
            c116 = c116,
            c117 = c117,
            c118 = c118,
            c119 = c119,
            c120 = c120,
            c121 = c121,
            c122 = c122,
            c123 = c123,
            c124 = c124,
            c125 = c125,
            c126 = c126,
            c127 = c127,
            c128 = c128,
            c130_pr = c130_pr,
            c130_cs = c130_cs,
            c131_pr = c131_pr,
            c131_cs = c131_cs,
            c132 = c132,
            c133 = c133,
            c134 = c134,
            c135 = c135,
            c139_ln = c139_ln,
            c139_xf = c139_xf,
            c140_ln = c140_ln,
            c140_xf = c140_xf,
            c141_ln = c141_ln,
            c141_xf = c141_xf,
            c144 = c144,
            c145 = c145,
            c146 = c146,
            c147 = c147,
            c148_ln = c148_ln,
            c148_xf = c148_xf,
            c149_ln = c149_ln,
            c149_xf = c149_xf,
            c150_ln = c150_ln,
            c150_xf = c150_xf,
            c151_ln = c151_ln,
            c151_xf = c151_xf,
            c156 = c156
        )

        if include_ctg
            vars = (;vars...,
            s_jtk_plus_ln = s_jtk_plus_ln,
            s_jtk_plus_xf = s_jtk_plus_xf,
            z_jtk_s_ln = z_jtk_s_ln,
            z_jtk_s_xf = z_jtk_s_xf,
            p_jtk_ln = p_jtk_ln,
            p_t_sl = p_t_sl,
            p_jtk_xf = p_jtk_xf,
            θ_itk = θ_itk,
            z_tk_ctg = z_tk_ctg,
            z_t_ctg_min = z_t_ctg_min,
            z_t_ctg_avg = z_t_ctg_avg)
            cons = (;cons...,
                    c4=c4, 
                    c5=c5, 
                    c10=c10, 
                    c158_ln=c158_ln, 
                    c158_xf=c158_xf, 
                    c159_ln=c159_ln, 
                    c159_xf=c159_xf, 
                    c160_ln=c160_ln, 
                    c160_xf=c160_xf, 
                    c161_ln=c161_ln, 
                    c161_xf=c161_xf, 
                    c162=c162, 
                    c163=c163)
        end

        core, vars2, cons2 = user_callback(core, vars, cons)
        model = ExaModel(core; prod = true, kwargs...)

        vars = (;vars..., vars2...)
        cons = (;cons..., cons2...)
   
    @info "built model"
    return model, cons, vars, lengths, sc_data_array

end

