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
`ifndef TB_QUANT
    `define TB_QUANT 8
`endif

module QUOTA_TB;
    // ---------------------------------------------------------------------------
    // Parameters for DUT instantiation (override with +define+TB_*)
    // ---------------------------------------------------------------------------
    localparam int TB_BITSTREAM  = `TB_BITSTREAM;
    localparam int TB_QUANT      = `TB_QUANT;

    // ---------------------------------------------------------------------------
    // DUT I/O
    // ---------------------------------------------------------------------------
    logic [TB_QUANT-1:0]             data;
    logic [$clog2(TB_BITSTREAM)-1:0] quota_out;
    // ---------------------------------------------------------------------------
    // DUT instantiation
    // ---------------------------------------------------------------------------
    QUOTA #(
        .BITSTREAM  (TB_BITSTREAM),
        .QUANT      (TB_QUANT)
    ) dut (
        .data   (data),
        .quota  (quota_out)
    );

    // ---------------------------------------------------------------------------
    // Reference model (mirrors the RTL intent)
    // u = q + 128
    // s = (u * T + (1<< (QUANT-1) ) ) >> QUANT
    // return s
    // ---------------------------------------------------------------------------
    function automatic logic [TB_QUANT-1:0] ref_quota (int q);
        logic [$clog2(TB_BITSTREAM)-1:0] tb_quota_out;
        begin
            int u = q + (1 << (TB_QUANT-1));
            int s = (u * TB_BITSTREAM + (1<< (TB_QUANT-1) ) ) >> TB_QUANT;
            tb_quota_out = s;
            return tb_quota_out;
        end
    endfunction




    integer errors = 0;
    integer DATA_RANGE_NEG = 1 - (1 << (TB_QUANT-1));
    integer DATA_RANGE_POS = (1 << (TB_QUANT-1));

    // Optional VCD/FSDB dump
    initial begin
    `ifdef DUMPVCD
        $dumpfile("quota_tb.vcd");
        $dumpvars(0, QUOTA_TB);
    `endif
    end

    // Main test sequence: sweep all q, then random samples
    initial begin
        $display("\n=== QUOTA TB start ===");
        $display("Params: BITSTREAM=%0d QUANT=%0d", TB_BITSTREAM, TB_QUANT);


        // Deterministic sweep
        for (int q = DATA_RANGE_NEG; q <= DATA_RANGE_POS; q++)begin
            logic [$clog2(TB_BITSTREAM)-1:0] golden;

            data = q;
            #1; // settle (purely combinational)
            golden = ref_quota(q);

            if (quota_out !== golden) begin
                errors++;
                $error("Mismatch at q=%0d: design=%0d tb=%0d", q, quota_out, golden);
            end
        end

        // A few randoms (in-range)
        for (int t = 0; t < 1000; t++) begin
            logic [$clog2(TB_BITSTREAM)-1:0] golden;
            int q = $urandom_range(DATA_RANGE_NEG, DATA_RANGE_POS);
            data = q;
            #1; // settle (purely combinational)
            golden = ref_quota(q);

            if (quota_out !== golden) begin
                errors++;
                $error("Mismatch at q=%0d: design=%0d tb=%0d", q, quota_out, golden);
            end
        end

        if (errors == 0)
            $display("\n[PASS] All checks passed.\n");
        else 
            $display("\n[FAIL] %0d mismatches detected.\n", errors);

        $fatal(1);
        $finish;
    end
endmodule
