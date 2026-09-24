# cost1
function gen_cost(g, pg)
    return g.c[1] * pg^2 + g.c[2] * pg + g.c[3]
end

function c_ref_angle_polar(va)
    return va
end

function c_to_active_power_flow_polar(b, p_f, vm_f, vm_t, va_f, va_t)
    return p_f - b.c5 * vm_f^2 -
    b.c3 * (vm_f * vm_t * cos(va_f - va_t)) -
    b.c4 * (vm_f * vm_t * sin(va_f - va_t))
end

function c_to_reactive_power_flow_polar(b, q_f, vm_f, vm_t, va_f, va_t)
    return q_f +
    b.c6 * vm_f^2 +
    b.c4 * (vm_f * vm_t * cos(va_f - va_t)) -
    b.c3 * (vm_f * vm_t * sin(va_f - va_t))
end

function c_from_active_power_flow_polar(b, p_t, vm_f, vm_t, va_f, va_t)
    return p_t - b.c7 * vm_t^2 -
    b.c1 * (vm_t * vm_f * cos(va_t - va_f)) -
    b.c2 * (vm_t * vm_f * sin(va_t - va_f))
end

function c_from_reactive_power_flow_polar(b, q_t, vm_f, vm_t, va_f, va_t)
    return q_t +
    b.c8 * vm_t^2 +
    b.c2 * (vm_t * vm_f * cos(va_t - va_f)) -
    b.c1 * (vm_t * vm_f * sin(va_t - va_f))
end

function c_phase_angle_diff_polar(b, va_f, va_t)
    return va_f - va_t
end

function c_active_power_balance_demand_polar(b, vm)
    return b.pd + b.gs * vm^2
end

function c_reactive_power_balance_demand_polar(b, vm)
    return b.qd - b.bs * vm^2
end

#no coordinates specified
function c_thermal_limit(b, p,q)
    return p^2 + q^2 - b.rate_a^2
end

#only for mp
function c_ramp(pg0, pg1)
    return pg0 - pg1
end

#rectangular constraints
function c_ref_angle_rect(vr, vim)
    return atan(vim/vr)
end

function c_to_active_power_flow_rect(b, p_f, vr_f, vr_t, vim_f, vim_t)
    return p_f - b.c5 * (vr_f^2+vim_f^2) -
    b.c3 * (vr_f*vr_t + vim_f*vim_t) -
    b.c4 * (vim_f*vr_t - vr_f*vim_t) 
end

function c_to_reactive_power_flow_rect(b, q_f, vr_f, vr_t, vim_f, vim_t)
    return q_f +
    b.c6 * (vr_f^2+vim_f^2) +
    b.c4 * (vr_f*vr_t + vim_f*vim_t) -
    b.c3 * (vim_f*vr_t - vr_f*vim_t)
end

function c_from_active_power_flow_rect(b, p_t, vr_f, vr_t, vim_f, vim_t)
    return  p_t - b.c7 * (vr_t^2+vim_t^2) -
    b.c1 * (vr_f*vr_t + vim_f*vim_t) -
    b.c2 * (vim_t*vr_f - vr_t*vim_f)
end

function c_from_reactive_power_flow_rect(b, q_t, vr_f, vr_t, vim_f, vim_t)
    return q_t +
        b.c8 * (vr_t^2+vim_t^2) +
        b.c2 * (vr_f*vr_t + vim_f*vim_t) -
        b.c1 * (vim_t*vr_f - vr_t*vim_f)
end

function c_phase_angle_diff_rect(b, vr_f, vr_t, vim_f, vim_t)
    return atan(vim_f/vr_f) - atan(vim_t/vr_t) 
end

function c_active_power_balance_demand_rect(b, vr, vim)
    return b.pd + b.gs * (vr^2 + vim^2)
end

function c_reactive_power_balance_demand_rect(b, vr, vim)
    return b.qd - b.bs * (vr^2 + vim^2)
end

function c_voltage_magnitude_rect(vr, vim)
    return vr^2 + vim^2
end

#storage constraints

#Formulation from Geth, Coffrin, Fobes (2020)
function c_active_stor_power(s, pst, pstd, pstc, I2)
    return pst + pstd - pstc - s.Pexts - s.Zr*I2
end

function c_active_storage_power_smooth(s, pst, pstd, I2)
    return pst + pstd - s.Pexts - s.Zr*I2
end

# DC storage: no current magnitude, so no resistive loss term. Everything else
# about the energy model is active power and carries over unchanged.
function c_active_stor_power_dc(s, pst, pstd, pstc)
    return pst + pstd - pstc - s.Pexts
end

function c_active_storage_power_smooth_dc(s, pst, pstd)
    return pst + pstd - s.Pexts
end

function c_reactive_stor_power(s, qst, qint, I2)
    return qst - qint - s.Qexts - s.Zim*I2
end

function c_ohms_polar(pst, qst, vm, I2)
    return pst^2 + qst^2 - (vm^2)*I2
end

function c_ohms_rect(pst, qst, vr, vim, I2)
    return pst^2 + qst^2 - (vr^2 + vim^2)*I2
end

function c_stor_state(s, E0, E1, pstc, pstd)
    return E0 - E1 - (s.charge_efficiency*pstc - pstd/s.discharge_efficiency)
end

function c_storage_state_smooth(s, E0, E1, discharge_func::Function, pstd)
    return E0 - E1 + discharge_func(pstd, s.thermal_rating)
end

function c_transfer_lim(s, pst, qst)
    return pst^2 + qst^2 - s.thermal_rating^2
end

#used for charge and discharge limits
function c_discharge_lim(pstd, pstc)
    return pstd - pstc
end

function c_discharge_limit_smooth(pstd)
    return pstd
end

function c_comp(pstd, pstc)
    return pstd*pstc
end

# DCOPF
#
# The susceptance is split out of the flow equation so a caller can compute it
# once, as data, and hand a MODIFIED value in. That is what makes a line outage
# expressible in DC: the AC outage zeroes the admittance coefficients c1..c8,
# but the DC flow is derived from `br_r`/`br_x`, and zeroing `br_x` gives 0/0 on
# the zero-resistance branches real cases contain (3 of 9 in case9, 9 of 186 in
# case118). Masking `bs` itself has no such hole.
dc_susceptance(br) = -br.br_x / (br.br_r^2 + br.br_x^2)

function c_ohms_law_dcopf(bs, pf, va_f, va_t)
    return -bs * (va_f - va_t) - pf
end

function c_active_power_balance_dc(b)
    return b.pd + b.gs
end

