// First undefine 'bgraph' so we can easily reload this file.
require.undef('bgraph');

define('bgraph', ['d3'], function (d3) {

    function draw(container, verticies, edges, traversal, supplementary_data, width, height) {
        width = width || 600;
        height = height || 200;
        var svg = d3.select(container).append("svg")
            .attr('width', width)
            .attr('height', height)
            .append("g");

        // TODO - create arrows: http://thenewcode.com/1068/Making-Arrows-in-SVG
        // svg.append("marker").attr("id", "arrowhead").attr("orient", "auto").append("polygon").attr("positions", "0 0, 10 3.5, 0 7")

        var text = d3.select(container).append("div")
            .attr("class", "edges").text("Edges: ")

        var lines = svg.selectAll('line').data(edges);
        lines.enter()
            .append('line')
            .attr("x1", function (d, i) { return verticies[d[0]]["position"][0]; })
            .attr("y1", function (d, i) { return verticies[d[0]]["position"][1]; })
            .attr("x2", function (d, i) { return verticies[d[1]]["position"][0]; })
            .attr("y2", function (d, i) { return verticies[d[1]]["position"][1]; })
            .style("stroke", "grey");

        var rr = 30;

        const traversal_edges = [];
        for (let i = 0; i < traversal.length - 1; i++) {
            traversal_edges.push([traversal[i][0], traversal[i + 1][0]])
        }

        var traversal_lines = svg.selectAll('line.trav').data(traversal_edges);
        traversal_lines.enter()
            .append('line')
            .attr('class', "trav")
            .attr("x1", function (d, i) { return verticies[d[0]]["position"][0]; })
            .attr("y1", function (d, i) { return verticies[d[0]]["position"][1]; })
            .attr("x2", function (d, i) { return verticies[d[1]]["position"][0]; })
            .attr("y2", function (d, i) { return verticies[d[1]]["position"][1]; })
            .style("stroke", "#C657E1");

        var circles = svg.selectAll('circle').data(Object.entries(traversal).filter(d => d[1][1] == 0));
        circles.enter()
            .append('circle')
            .attr("cx", function (d, i) { return verticies[d[1][0]]["position"][0]; })
            .attr("cy", function (d, i) { return verticies[d[1][0]]["position"][1]; })
            .attr("r", 20)
            .attr("class", "nohighlighted")
            .style("opacity", 0.9)

        traversal.forEach( d => {
            if (d[1] == "-1") {
                svg.append('circle')
                    .attr('cx', verticies[d[0]]["position"][0])
                    .attr('cy', verticies[d[0]]["position"][1])
                    .attr('r', 20)
                    .attr('class', "highlighted")
                    .style("opacity", 0.9)
            }
        })

        var images = svg.selectAll("image").data(Object.entries(verticies));

        var sep = "),(";

        var zoomin_size = 300;

        images.enter()
            .append("svg:image")
            .attr('x', function (d, i) { return d[1]["position"][0] - rr / 2; })
            .attr("y", function (d, i) { return d[1]["position"][1] - rr / 2; })
            .attr("width", rr)
            .attr("height", rr)
            .attr("node_name", function (d, i) { return d[0]; })
            .attr("xlink:href", function (d, i) { return "data:image/png;base64," + d[1]["images"][0]; })
            .on('click', function (d, i) {
                // d3.select("#xyzxyz").attr("xlink:href", d3.select(this).attr("xlink:href"));
                const images_popup = svg.append("svg")
                    .attr("x", d3.select(this).attr("x"))
                    .attr("y", d3.select(this).attr("y"))
                    .on('click', function () {
                        d3.select(this).remove();
                    });
                var x_pos = 0
                var deltaX, deltaY;
                var dragHandler = d3.drag()
                    .on("start", function () {
                        var current = d3.select(this);
                        deltaX = current.attr("x") - d3.event.x;
                        deltaY = current.attr("y") - d3.event.y;
                    })
                    .on("drag", function () {
                        d3.select(this)
                            .attr("x", d3.event.x + deltaX)
                            .attr("y", d3.event.y + deltaY);
                    });
                dragHandler(images_popup);
                d[1]["images"].forEach(elm => {
                    images_popup.append("svg:image")
                        .attr("width", zoomin_size)
                        .attr("height", zoomin_size)
                        .attr("x", x_pos)
                        .attr("y", 0)
                        .attr("xlink:href", "data:image/png;base64," + elm);
                    x_pos += zoomin_size;
                });
                var node_name = d3.select(this).attr("node_name")
                var text_box = images_popup.append("text")
                    .attr("x", x_pos+10)
                    .attr("y", 30)
                text_box.append("tspan").text(node_name)
                if (node_name in supplementary_data) {
                    supplementary_data[node_name].forEach(elem => {
                        text_box.append("tspan").text(elem)
                            .attr("dy", "1.2em")
                            .attr("x", x_pos+10)
                    })
                }
                d3.select(container).select("div.edges")
                    .text(d3.select(container).select("div.edges").text() + sep + d3.select(this).attr("node_name"));
                if (sep == ",") {
                    sep = "),("
                } else {
                    sep = ","
                };
            })
            .on('mouseout', function () {
                d3.select(this)
                    .transition('fade').duration(500)
                    .attr("width", rr)
                    .attr("height", rr);
            });
    }
    return draw;
});

element.append('<small>&#x25C9; &#x25CB; &#x25EF; Loaded bgraph.js &#x25CC; &#x25CE; &#x25CF;</small>');
