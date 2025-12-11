`timescale 1ns/1ps

module tb_fifo;
    localparam int DATA_WIDTH    = 64;
    localparam int DEPTH         = 2;

    logic clk, rst_n;
    logic w_valid;
    logic r_ready;
    logic [DATA_WIDTH-1 : 0]  data_in;
    wire  [DATA_WIDTH-1 : 0]  data_out;
    wire fifo_full, fifo_empty;
    int i,j;

    fifo
    #(  .WIDTH      (DATA_WIDTH),
        .DEPTH      (DEPTH)
    )   inst_fifo
    (
        .clk         (clk),
        .rst_n       (rst_n),
        .fifo_full   (fifo_full),
        .fifo_empty  (fifo_empty),
        .w_valid     (w_valid),
        .r_ready     (r_ready),
        .data_in     (data_in),
        .data_out    (data_out)
    );

    // clock -----------------------------------------------------------
    always #50 clk = ~clk;

    // *****************************************************************
    // Create TB FIFO for check : 
    // *****************************************************************   
    localparam int TB_DEPTH         = 1024;
    localparam int TB_PTR_NUM_BITS  = $clog2(TB_DEPTH);

    logic [DATA_WIDTH-1:0] fifo_mem [0:TB_DEPTH-1];
    logic [TB_PTR_NUM_BITS:0] wrp, rdp;
    logic [TB_PTR_NUM_BITS:0] drp;  
    logic close_push, close_pop;
    logic [DATA_WIDTH-1:0] push_data;
    logic fifo_push, fifo_pop;
    logic [DATA_WIDTH-1:0] push_data_wire;

    assign fifo_push = !fifo_full  & !close_push;
    assign fifo_pop  = !fifo_empty & !close_pop;

    // pointer & depth counter -----------------------------------------
    always_ff @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            drp <= '0;
            rdp <= '0;
            wrp <= '0;
        end
        else
        begin
            if(fifo_push)
            begin
                fifo_mem[wrp] <= push_data_wire;
                if(wrp < TB_DEPTH-1)
                    wrp <= wrp + 1;
                else
                    wrp <= '0;
                drp <= drp + 1;
            end
            if(fifo_pop)
            begin
                if(rdp < TB_DEPTH-1)
                    rdp <= rdp + 1;
                else
                    rdp <= '0;
                drp <= drp - 1;
                auto_check;
            end
        end
    end

    // *****************************************************************
    // task generate_data : 
    //  - input data and notify to fifo full
    //  - it will reset valid at next clock if push to fifo.
    //  - it will also to push in TB FIFO.
    // *****************************************************************   
    task automatic generate_data;
        input  logic [DATA_WIDTH-1 : 0] data;
        begin
            push_data = data;
            @(posedge clk);
        end
    endtask

    // *****************************************************************
    // task pop_write_mem : 
    //  - create a exp memory and design memory for check.
    //  - pop data and notify to fifo empty.
    //  - it will reset ready at next clock if push to fifo.
    //  - it will also to pop in TB FIFO.
    //  - !! mem_addr only plus one at pop fifo.
    // *****************************************************************   

    // combinational connection for tb -> DUT & tb fifo mem
    always_comb begin
        push_data_wire = push_data;
        data_in        = push_data_wire;
        r_ready        = fifo_pop;
        w_valid        = fifo_push;
    end

    // *****************************************************************
    // task close_push_pop : 
    //  - a switch can control the push / pop ------- [1] : push / [0] : pop
    //  - compare exp and design mem result
    //  - use mem_addr pointer to know the mem data nums.
    // *****************************************************************   
    task automatic close_push_pop;
        input  logic [1:0] switch;
        begin
            close_push = !switch[1]; 
            close_pop  = !switch[0];
        end
    endtask

    wire [DATA_WIDTH-1:0] exp_data;
    assign exp_data = fifo_mem[rdp];

    // *****************************************************************
    // task auto_check : 
    //  - compare exp and design mem result
    // *****************************************************************   
    task automatic auto_check;
        begin
            if(exp_data == data_out)
                $display("  Correct : %h" , exp_data);
            else
            begin
                $display("    ERROR : auto check compare result is failed, exp_data : %h , data_out : %h." , exp_data, data_out);
                $stop;
            end
        end
    endtask

    // *****************************************************************
    // task random_data_generate : 
    //  - double layer for loop
    //      - total_loop for : total generate and pop to mem data cycle.
    //          - random for generate   : total generate data.
    //          - random for pop to mem : total pop data. 
    // *****************************************************************   
    task automatic random_data_generate;
        input int unsigned total_loop;
        begin
            for(i = 0 ; i < total_loop ; i = i+1)
            fork
                close_push_pop(({$random} % 2'b11));
                for(j = 0 ; j < ({$random} % (2 * DEPTH)) ; j = j+1) 
                    generate_data({$random} % 32'hFFFFFFFF);
            join
        end
    endtask


    // *****************************************************************
    // - TB Need test case
    //  - 1. check fifo empty (no w_valid)      (OK)
    //  - 2. check fifo full (control r_ready)  (OK)
    //  - 3. basic auto check compare data :    (OK)
    //      - TB has a fifo and sram (behavior)
    //      - when data is generated, push to TB fifo and push to design fifo. 
    //      - design fifo and TB fifo have the same depth fifo layer.
    //  - 4. random test                        (OK)
    // *****************************************************************  
    initial 
    begin
        $dumpfile ("./tb_fir.vcd");
        $dumpvars (0, tb_fifo);

        // reset --------------------------------------------------------
        close_push_pop(2'b00);
        clk     = 0;
        rst_n   = 1;
        repeat(1) @(posedge clk)
        rst_n   = 0;
        repeat(1) @(posedge clk)
        rst_n   = 1;

        //  1. fifo empty testing ---------------------------------------
        if(!fifo_empty)
        begin
            $display ("    ERROR : fifo_empty is not working.");
            $stop;
        end

        //  2. fifo full testing ---------------------------------------
        close_push_pop(2'b10);
        for(i = 0; i < DEPTH ; i = i+1)
        begin
            if(fifo_full) 
            begin
                $display("    ERROR : fifo_full is not match to depth.");
                $stop;
            end
            generate_data(i);
        end
        generate_data(i);
        @(posedge clk);
        if(!fifo_full) 
        begin
            $display("    ERROR : fifo_full is not working.");
            $stop;
        end

        //  3. basic auto check ----------------------------------------
        close_push_pop(2'b11);
        for(i = 0; i < 100; i= i+1)  
            generate_data(i);

        //  4. random task ---------------------------------------------
        random_data_generate(600);

        // finish ------------------------------------------------------
        repeat(5) @(posedge clk);
        $display ("CORRECT : Not have any error.");
        $finish;
    end

endmodule
/*
("    ERROR : fifo_empty is not working.");
("  CORRECT : Not have any error.");
("FIFO FULL : data %h is not input fifo." , data);
("    ERROR : fifo_full is not match to depth.");
("    ERROR : fifo_full is not working.");
("FIFO Empty: no data in fifo.")
("    ERROR : auto check compare result is failed, in memory address : %d." , i);
*/
