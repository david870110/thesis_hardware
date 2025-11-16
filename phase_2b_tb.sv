`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// Testbench: weyl_tb
// Purpose  : Self-checking, parameterized SV testbench for module WEYL
// Design   : expects module WEYL in your compile list (e.g., weyl_table.sv)
// Params   : Override at compile time with +define+TB_BITSTREAM=... etc.
// Sim tips :
//   - Questa  : vlog -sv weyl_table.sv weyl_tb.sv ; vsim -c weyl_tb -do "run -all;
//    quit"
//   - VCS     : vcs -sverilog weyl_table.sv weyl_tb.sv -R
//   - Verilator: verilator -sv --cc --exe sim_main.cpp weyl_table.sv weyl_tb.sv  
//   (or --binary)
//   - Icarus  : iverilog -g2012 -o tb.vvp weyl_table.sv weyl_tb.sv ; vvp tb.vvp
// -----------------------------------------------------------------------------

`ifndef TB_BITSTREAM
    `define TB_BITSTREAM 64
`endif


module PHASE_2b_TB;
    // ---------------------------------------------------------------------------
    // Parameters for DUT instantiation (override with +define+TB_*)
    // ---------------------------------------------------------------------------
    localparam int TB_BITSTREAM  = `TB_BITSTREAM;

    // ---------------------------------------------------------------------------
    // DUT I/O
    // ---------------------------------------------------------------------------
    logic [1:0]              k;
    logic [TB_BITSTREAM-1:0] iBitstream;
    logic [TB_BITSTREAM-1:0] oBitstream;
    // ---------------------------------------------------------------------------
    // DUT instantiation
    // ---------------------------------------------------------------------------
    PHASE_2b #(
        .BITSTREAM  (TB_BITSTREAM)
    ) dut (
        .k          (k),
        .in_bits    (iBitstream),
        .out_bits   (oBitstream)
    );
    // ---------------------------------------------------------------------------
    // Reference model (mirrors the RTL intent)
    // u = q + 128
    // s = (u * T + (1<< (QUANT-1) ) ) >> QUANT
    // return s
    // ---------------------------------------------------------------------------
    function automatic logic [63:0] rand64();
        return { $urandom(), $urandom() };  // 32b + 32b = 64b
    endfunction


    function automatic logic [TB_BITSTREAM-1:0] ref_phase (
        input logic [TB_BITSTREAM-1:0]  in_bits,
        input int                       k
    );
        logic [TB_BITSTREAM-1:0] out_bits;
        int km;
        begin
            km = ((k % TB_BITSTREAM) + TB_BITSTREAM) % TB_BITSTREAM;  // 安全取 mod，避免負值
            out_bits = (km == 0) ? in_bits
                                : ((in_bits >> km) | (in_bits << (TB_BITSTREAM - km))); // 右旋 k
            return out_bits;
        end
    endfunction




 
    integer errors = 0;
    integer k_phase;

    // Optional VCD/FSDB dump
    initial begin
    `ifdef DUMPVCD
        $dumpfile("phase_2b_tb.vcd");
        $dumpvars(0, PHASE_2b_TB);
    `endif
    end

    // Main test sequence: sweep all q, then random samples
    initial begin
        $display("\n=== PHASE TB start ===");
        $display("Params: BITSTREAM=%0d ", TB_BITSTREAM);


        // Deterministic sweep
        for (int q = 0; q <= 100; q++)begin
            logic [TB_BITSTREAM-1:0] golden;
            k_phase = q % 4;
            iBitstream = rand64();
            k = (k_phase);
            #1; // settle (purely combinational)
            golden = ref_phase(k_phase,rand64());

            if (oBitstream !== golden) begin
                errors++;
                $error("Mismatch at q=%0d: design=%0d tb=%0d", q, oBitstream, golden);
            end
        end

        // // A few randoms (in-range)
        // for (int t = 0; t < 1000; t++) begin
        //     logic [$clog2(TB_BITSTREAM)-1:0] golden;
        //     int q = $urandom_range(DATA_RANGE_NEG, DATA_RANGE_POS);
        //     data = q;
        //     #1; // settle (purely combinational)
        //     golden = ref_quota(q);

        //     if (quota_out !== golden) begin
        //         errors++;
        //         $error("Mismatch at q=%0d: design=%0d tb=%0d", q, quota_out, golden);
        //     end
        // end

        if (errors == 0)begin
            $display("\n[PASS] All checks passed.\n");
            $finish;
        end
        else begin
            $display("\n[FAIL] %0d mismatches detected.\n", errors);
            $fatal(1);
        end
    end
endmodule
