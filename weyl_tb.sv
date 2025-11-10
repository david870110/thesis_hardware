`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Testbench: weyl_tb
// Purpose  : Self-checking, parameterized SV testbench for module WEYL
// Design   : expects module WEYL in your compile list (e.g., weyl_table.sv)
// Params   : Override at compile time with +define+TB_BITSTREAM=... etc.
// Sim tips :
//   - Questa  : vlog -sv weyl_table.sv weyl_tb.sv ; vsim -c weyl_tb -do "run -all; quit"
//   - VCS     : vcs -sverilog weyl_table.sv weyl_tb.sv -R
//   - Verilator: verilator -sv --cc --exe sim_main.cpp weyl_table.sv weyl_tb.sv  (or --binary)
//   - Icarus  : iverilog -g2012 -o tb.vvp weyl_table.sv weyl_tb.sv ; vvp tb.vvp
// -----------------------------------------------------------------------------

`ifndef TB_BITSTREAM
    `define TB_BITSTREAM 64
`endif
`ifndef TB_BASE
    `define TB_BASE 61
`endif
`ifndef TB_STRIDE
    `define TB_STRIDE 17
`endif

module weyl_tb;
    // ---------------------------------------------------------------------------
    // Parameters for DUT instantiation (override with +define+TB_*)
    // ---------------------------------------------------------------------------
    localparam int TB_BITSTREAM = `TB_BITSTREAM;
    localparam int TB_BASE      = `TB_BASE;
    localparam int TB_STRIDE    = `TB_STRIDE;

    // ---------------------------------------------------------------------------
    // DUT I/O
    // ---------------------------------------------------------------------------
    logic [$clog2(TB_BITSTREAM)-1:0] quota_num;
    logic [TB_BITSTREAM-1:0]         weyl_out;

    // ---------------------------------------------------------------------------
    // DUT instantiation
    // ---------------------------------------------------------------------------
    WEYL #(
        .BITSTREAM (TB_BITSTREAM),
        .BASE      (TB_BASE),
        .STRIDE    (TB_STRIDE)
    ) dut (
        .quota_num (quota_num),
        .weyl_out  (weyl_out)
    );

    // ---------------------------------------------------------------------------
    // Reference model (mirrors the RTL intent)
    // Behavior inferred from WEYL:
    //   one_hot = 1 << quota_num
    //   quota_mask[0] = 0; for k>=1: quota_mask[k] = quota_mask[k-1] | one_hot[k-1]
    //   weyl_out[(BASE + i*STRIDE) % BITSTREAM] = quota_mask[i]
    // ---------------------------------------------------------------------------
    function automatic logic [TB_BITSTREAM-1:0] ref_weyl (int q);
        logic [TB_BITSTREAM-1:0] one_hot;
        logic [TB_BITSTREAM-1:0] quota_mask;
        logic [TB_BITSTREAM-1:0] permuted;
        int idx;
        begin
            one_hot    = '0;
            quota_mask = '0;
            permuted   = '0;

        for (int i = 0; i < q; i++) begin
            idx = (TB_BASE + TB_STRIDE) % TB_BITSTREAM;
            permuted[idx] = 1'b1;
        end 
        return permuted;
        end
    endfunction




    // Small util: gcd, to warn if STRIDE not co-prime with BITSTREAM
    function automatic int gcd (int a, int b);
        int t;
        begin
            a = (a < 0) ? -a : a; 
            b = (b < 0) ? -b : b;
            if (a == 0) return b; 
            if (b == 0) return a;
            while (b != 0) begin 
                t = a % b; 
                a = b; 
                b = t; 
                end
            return a;
        end
    endfunction



    // Coverage (sampled on demand)
    covergroup cg_weyl;
        coverpoint quota_num {
            bins all_vals[] = {[0:$clog2(TB_BITSTREAM)-1]};
        }
        coverpoint $countones(weyl_out) {
            bins zeros_to_bs[] = {[0:TB_BITSTREAM]};
        }
    endgroup

    cg_weyl cov = new();

    integer errors = 0;

    // Optional VCD/FSDB dump
    initial begin
    `ifdef DUMPVCD
        $dumpfile("weyl_tb.vcd");
        $dumpvars(0, weyl_tb);
    `endif
    end

    // Main test sequence: sweep all q, then random samples
    initial begin
        $display("\n=== WEYL TB start ===");
        $display("Params: BITSTREAM=%0d BASE=%0d STRIDE=%0d", TB_BITSTREAM, TB_BASE, TB_STRIDE);

        if (gcd(TB_BITSTREAM, TB_STRIDE) != 1)
        $display("[WARN] STRIDE and BITSTREAM are not co-prime; permutation will repeat every %0d.", TB_BITSTREAM / gcd(TB_BITSTREAM, TB_STRIDE));

        // Deterministic sweep
        for (int q = 0; q < $clog2(TB_BITSTREAM) ; q++) begin
            logic [TB_BITSTREAM-1:0] golden;

            quota_num = q;
            #1; // settle (purely combinational)
            golden = ref_weyl(q);

            if (weyl_out !== golden) begin
                errors++;
                $error("Mismatch at q=%0d: got=%0h exp=%0h", q, weyl_out, golden);
            end
            cov.sample();
        end

        // A few randoms (in-range)
        for (int t = 0; t < 100; t++) begin
            logic [TB_BITSTREAM-1:0] golden;
            int q = $urandom_range(0, $clog2(TB_BITSTREAM)-1);
            quota_num = q;
            #1;
            golden = ref_weyl(q);
            if (weyl_out !== golden) begin
                errors++;
                $error("[RND] Mismatch at q=%0d: got=%0h exp=%0h", q, weyl_out, golden);
            end
            cov.sample();
        end

        if (errors == 0)
            $display("\n[PASS] All checks passed.\n");
        else 
            $display("\n[FAIL] %0d mismatches detected.\n", errors);

        $fatal(1);
        $finish;
    end
endmodule
