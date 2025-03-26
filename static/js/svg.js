
class MySVG 
{
  constructor(wrapper, tooltip) 
  {
    this.wrapper = document.getElementById(wrapper);
    this.svg = this.config_svg();
    this.call_back = { start: function () { }, draw: function () { }, end: function () { } };
    this.tooltip = tooltip;
    this.selected_items = [];
  }
  on(type, call_back) 
  {
    if (!arguments.length)
      return this.call_back;
    if (arguments.length === 1)
      return this.call_back[type];
    if (Object.keys(this.call_back).indexOf(type) > -1)
      this.call_back[type] = call_back;

    return this;
  }
  config_svg() 
  {
    var svg = d3.select(this.wrapper).select("svg");

    if (svg.empty())
      svg = d3.select(this.wrapper).append("svg");
    else
      svg.selectAll("*").remove();

    svg
      .attr("width", this.wrapper.clientWidth)
      .attr("height", this.wrapper.clientHeight)
      .append("g");

    return svg;
  };
  draw(data, data_summary, palette) 
  {
    throw new Error('You have to implement the method draw!');
  }
  select(data, redraw = true) 
  {
    throw new Error('You have to implement the method select!');
  }
}

class TokenInfo
{
  get_table(data, count, show_id = true)
  {
    var html = "";

    if(show_id)
      html  += "<p><span class='font-weight-bold'>Token: </span>" + data.id + "</p>";

    html += "<p><span class='font-weight-bold'>Entries: </span>" + count + "</p>";          

    html += "<div class='table-responsive'>";
    html += "<table class='table table-bordered' id='dataTable' width='100%' cellspacing='0'>";
    html += "<thead><tr><th>word</th><th>POS-tag</th><th>Entity</th></tr></thead>";
    html += "<tbody>";
    var aux_search = [];

    for(let i = 0; i < data.word.length; i++)
    {
      var search_key = data.word[i] + "," + data.postag[i][1] + "," + (data.named_entity[i] === null ? "" : data.named_entity[i]);
    
      if( aux_search.indexOf(search_key) === -1 )
      {  
        aux_search.push(search_key);
        
        html += "<tr>";

        var text = data.word[i];

        var id     = data.id.trim()
        var index  = text.toLowerCase().indexOf(id.toLowerCase());
        var before = text.slice(0, index);
        var after  = text.slice(index + id.length);
        var token  = text.substring(index, index + id.length);        

        html += "<td>" + before + "<span class='word-part'>" + token + "</span>" + after + "</td>";
        html += "<td>" + data.postag[i][1] + "</td>";
        html += "<td>" + (data.named_entity[i] === null ? "" : data.named_entity[i]) + "</td>";
        html += "</tr>";
      }
    }

    html += "</tbody>";
    html += "</table>"
    html += "</div>";   
    
    return html;
  }
}

class ScatterPlot extends MySVG 
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.legend = null;
    this.circle_size = 3;
    this.selected_class = [];
  }
  config_legend() 
  {
    if(this.legend === null)
      this.legend = d3.select(this.wrapper)
        .append("svg")
        .attr("width", this.wrapper.clientWidth)
        .attr("height", 20);
    else
      this.legend.selectAll("*").remove();

    this.legend.append("g");
  }
  draw(data, data_summary, palette) 
  {
    var _this = this;
    var label_list = [];
    this.svg = this.config_svg();

    var group = this.svg.select("g").attr("class", "scatter-plot");
    var xScale = d3.scaleLinear([data_summary.min_x, data_summary.max_x], [this.circle_size, this.svg.attr("width") - this.circle_size]);
    var yScale = d3.scaleLinear([data_summary.max_y, data_summary.min_y], [this.circle_size, this.svg.attr("height") - this.circle_size]);
    var circle = group.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", function (d) { return xScale(d.x); })
      .attr("cy", function (d) { return yScale(d.y); })
      .attr("r", this.circle_size)
      .attr("class", function (obj) { return _this.get_circle_class(obj); })    
      .style("fill", function (d) 
      { 
        if(label_list.indexOf(d.label) == -1)
          label_list.push(d.label); 
        
        return palette(d.label); 
      });

    let lassoBrush = lasso()
      .items(group.selectAll("circle"))
      .targetArea(this.svg)
      .on("end", function () 
      {
        var ids = lassoBrush.selectedItems()["_groups"][0]
          .map(function(item, i) { return item.__data__.sentence_id; });
        _this.call_back.end(ids, "lasso");
      });
    this.svg.call(lassoBrush);
    this.draw_legend(label_list, palette);
  }
  draw_legend(label_list, palette) 
  {
    label_list.sort();
    this.config_legend();

    var rect_size = 15;
    var legend_group = this.legend.select("g");   
    var _this = this;

    legend_group.selectAll("rect")
      .data(label_list)
      .enter()
      .append("rect")
      .attr("class", function (obj) { return _this.get_legend_class(obj); })    
      .attr('x', function (d, index) { return index * rect_size; })
      .attr('y', 0)
      .attr('rx', 2)
      .attr('ry', 2)
      .attr('width', rect_size)
      .attr('height', rect_size)
      .style("fill", function (d) { return palette(d); })
      .on("mouseover", function (event, target) { _this.tooltip.show(target, [event.clientX, event.clientY]); })
      .on("mouseout", function (event, target) { _this.tooltip.hide(); })
      .on("click", function (event, target) 
      {
        _this.selected_class = [];
        var text_ids = [];
        var selected = d3.select(this).classed("scatter-legend-selected");
 
        d3.select(this).classed("scatter-legend-selected", !selected);
        _this.legend.selectAll(".scatter-legend-selected").each(function(class_) { _this.selected_class.push(class_);  }) 

        if(_this.selected_class.length > 0)
          _this.svg.selectAll(".scatter-circle").each(function(obj)
          {
            if(_this.selected_class.indexOf(obj.label) !== -1)
              text_ids.push(obj.sentence_id);
          });

        text_ids = _this.call_back.end(text_ids, "class");
        _this.select(text_ids);
      });
  }
  get_legend_class(obj)
  {
    var class_name = "scatter-legend";

    if(this.selected_class.length > 0 && this.selected_class.indexOf(obj) !== -1)
      class_name += " scatter-legend-selected";

    return class_name;
  }
  get_circle_class(obj)
  {
    var class_name = "scatter-circle";

    if ((this.selected_items.length > 0 && this.selected_items.indexOf(obj.sentence_id) === -1) ||
        (this.selected_class.length > 0 && this.selected_class.indexOf(obj.label) === -1))
      class_name += " scatter-unselected";

    return class_name;    
  }
  select(data, redraw = true) 
  {
    var _this = this;
    this.selected_items = data;

    if(redraw)
    {
      this.selected_class = [];
      this.legend.selectAll(".scatter-legend-selected").each(function(class_) { _this.selected_class.push(class_);  })

      this.svg.select("g").selectAll("circle")
        .attr("class", function (obj) { return _this.get_circle_class(obj); });    
    }  
  }
}

class TreeMap extends MySVG 
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.treemap = null;
    this.token_info = new TokenInfo();
  }
  draw(data, data_summary, palette) 
  {
    this.svg = this.config_svg();
    this.group = this.svg.select("g");
    this.data = data;

    var root = d3.hierarchy(data)
      .count();
    this.treemap = d3.treemap().size([+this.svg.attr("width"), +this.svg.attr("height")]).round(true).paddingInner(1)
      .tile(d3.treemapBinary);
    var clusters = this.treemap(root)
      .descendants()
      .filter(function (obj) { return obj.height == 1; });

    this.update_rec(clusters);
  }
  update_rec(nodes) 
  {
    var _this = this;
    var rect = this.group.selectAll("rect").data(nodes, function (d) { return d.data.id; });

    rect.exit().remove();

    rect.enter()
      .append("rect")
      .attr("id", function (obj) { return obj.data.id; })
      .attr("class", function (obj) { return _this.get_rec_class(obj, this); })
      .attr('width', function (obj) { return obj.x1 - obj.x0; })
      .attr('height', function (obj) { return obj.y1 - obj.y0; })
      .attr("transform", function (obj) { return "translate(" + obj.x0 + "," + obj.y0 + ")"; })
      .on("click", function (event, target) 
      {
        var sentence_ids = [];
        var position = [];

        if(event.ctrlKey) 
        {
          //target (leaf), target.parent (culster), target.parent.parent (root)
          if (target.children === undefined)
            _this.update_rec(target.parent.parent.children);
          //target (cluster), target.children (leaft)
          else 
          {
            var new_target = Object.assign({}, target);
            new_target = Object.setPrototypeOf(new_target, Object.getPrototypeOf(target));
            new_target.depth -= 1;

            for (var i = 0; i < new_target.children.length; i++)
              new_target.children[i].depth -= 1;

            new_target = _this.treemap(new_target);
            _this.update_rec(new_target.children);
          }
        }
        else 
        {
          var selected = d3.select(event.target).classed("treemap-cluster-selected");

          _this.group.selectAll("rect").classed("treemap-cluster-selected", false);

          if (!selected) 
          {
            //target (leaf), target.parent (culster), target.parent.parent (root)
            if (target.children === undefined)
            {
              sentence_ids = sentence_ids.concat(target.data.sentences);
              position = position.concat(target.data.position);
            }
            //target (cluster), target.children (leaft)
            else
              target.children.forEach(function (obj, index)
              {
                sentence_ids = sentence_ids.concat(obj.data.sentences);
                position = position.concat(obj.data.position);
              });

            d3.select(event.target).classed("treemap-cluster-selected", true);
          }
        }

        _this.call_back.end(sentence_ids, position);
      })
      .on("mouseover", function (event, target) 
      {
        var msg = "";

        if (target.children === undefined)
        { 
          var count_sentences = _this.selected_items.length === 0 ? target.data.sentences.length :  _this.count(_this.selected_items, target);
          msg = _this.token_info.get_table(target.data, count_sentences);
        }  
        else
        {
          var count_token = _this.selected_items.length === 0 ? target.children.length : _this.count(_this.selected_items, target); 
          msg  = "<p><span class='font-weight-bold'>ID: </span>" + target.data.id + "</p>";
          msg += "<p><span class='font-weight-bold'>Count tokens: </span>" + count_token + "</p>";
          msg += "<p><span class='font-weight-bold'>Main tokens: </span><span>" + target.data.main_token + "</span></p>";
        }  

        _this.tooltip.show(msg, [event.clientX, event.clientY]);
      })
      .on("mouseout", function (event, target) { _this.tooltip.hide(); });
  }
  count(selected_items, obj)
  {
    //Process leaves
    if (obj.children === undefined)
      return obj.data.sentences
        .filter(function (sentence_id, j) { return selected_items.indexOf(sentence_id) !== -1; })
        .length;
    //Process intern nodes (clusters)    
    else
      return obj.children
        .filter(function (token, j) 
        {
          return token.data.sentences
            .filter(function (sentence_id, k) { return selected_items.indexOf(sentence_id) !== -1; })
            .length > 0;
        })
        .length;
  }
  get_rec_class(obj, dom_obj)
  {
    var class_name = obj.children === undefined ? "treemap-rect treemap-leaf" : "treemap-rect treemap-cluster";

    if (this.selected_items.length > 0)
    {  
      var original_length = obj.children === undefined ? obj.data.sentences.length : obj.children.length;
      var new_length = this.count(this.selected_items, obj)

      if(new_length == 0)
        class_name += " treemap-unselected";
      else if(obj.children !== undefined && original_length !== new_length)
        class_name += " treemap-partial-selected";
    }
    
    if(d3.select(dom_obj).classed("treemap-cluster-selected"))
      class_name += " treemap-cluster-selected"

    return class_name;    
  }
  select(data, redraw = true) 
  {
    this.selected_items = data;

    if(redraw)
    {
      var _this = this;
      this.svg.select("g").selectAll("rect")
        .attr("class", function (obj) { return _this.get_rec_class(obj, this); });
    }  
  }  
}

class WordCloud extends MySVG
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.token_info = new TokenInfo();
    this.selected_items = [];
  }  
  draw(data, data_summary, palette) 
  {
    var _this = this;
    this.svg = this.config_svg();
    var group = this.svg.select("g");
    var size_scale = d3.scaleLinear().domain([data_summary.min_freq, data_summary.max_freq]).range([20, 40]);
    var color_scale = d3.scaleSequential(d3.interpolate("#c8eac8", "#379337")).domain([data_summary.min_freq, data_summary.max_freq]);        

    var layout = d3.layout.cloud()
    .size([+this.svg.attr("width"), +this.svg.attr("height")])
    .words(data)
    .padding(5)
    .font("sans-serif")
    .fontSize(function(d) { return size_scale(d.frequency); })
    .on("end", function(words)
    {
      group
        .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
        .selectAll("text")
        .data(words)
        .enter()
        .append("text")
        .style("font-size", function(d) { return d.size + "px"; })
        .style("font-family", function(d) { return d.font; })
        .style("fill", function(d) { return color_scale(d.frequency); })
        .attr("class", "word")
        .attr("text-anchor", "middle")
        .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
        .text(function(d) { return d.id; })
        .on("click", function(event, target)
        {
          var selected = d3.select(event.target).classed("word-selected");
          var sentences = [];
          var position = [];
          group.selectAll("text").classed("word-selected", false);

          if(!selected)  
          {
            d3.select(event.target).classed("word-selected", true);
            sentences = target.sentences;
            position = target.position;
          }  

          _this.call_back.end(sentences, position);
        })
        .on("mouseover", function(event, target)
        {
          var msg = _this.token_info.get_table(target, target.frequency, false);
          _this.tooltip.show(msg, [event.clientX, event.clientY]);
        })                
        .on("mouseout", function(event, target) { _this.tooltip.hide(); });
    });
    layout.start();    
  }
  select(data, redraw = true) 
  {
    this.selected_items = data;
  }    
}

class SankyDiagram extends MySVG 
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.group = null;
    this.token_info = new TokenInfo();
    this.selected_items = [];
    this.selected_classes = [];
    this.clicked_class = [];
    this.stn2class = null;
    this.token_node_color = "#91b691";
    this.margin = {top: 5, bottom: 5, left: 5, right: 5, text_offset: 6};
  }
  update_link(links)
  {
    var _this = this;
    var lines = this.group
      .append("g")
      .attr("id", "links")
      .selectAll(".sunkey-link")
      .data(links, function(d) { return  d.source.id + "_" + d.source.type + ":" + d.target.id + "_" + d.target.type; });

    lines.exit().remove();

    lines.enter()
      .append("path")
      .attr("class", function (obj) { return _this.get_link_class(obj); })
      .attr("d", d3.sankeyLinkHorizontal())
      .attr("stroke-width", function(d) { return d.width; })
      .on("mouseover", function (event, target) 
      {
        var msg = "<p><span class='font-weight-bold'>Mean abs. score: </span>" + target.value.toFixed(5) + "</p>";
       _this.tooltip.show(msg, [event.clientX, event.clientY]);

      })
      .on("mouseout", function (event, target) { _this.tooltip.hide(); });            
  }
  update_node(sankey, nodes, palette)
  {
    var _this = this;
    var node = this.group
      .append("g")
      .attr("id", "nodes")
      .selectAll(".sunkey-node")
      .data(nodes, function(d){ return d.id + "_" + d.type; })

    // node.exit().remove();

    .enter()
      .append("g")
      .attr("class", "sunkey-node");  

    node.append("rect")
      .attr("class", function (obj) { return _this.get_node_class(obj) })
      .attr("x", function(d) { return d.x0; })
      .attr("y", function(d) { return d.y0; })
      .attr("height", function(d) { return d.y1 - d.y0; })
      .attr("width", sankey.nodeWidth())
      .style("fill", function(d) { return d.type == "class" ? d.color = palette(d.id) : d.color = _this.token_node_color; })
      .on("click", function (event, target) 
      {
        var sentence_ids = [];
        var position = [];
        _this.clicked_class = [];
        
        var selected = d3.select(event.target).classed("sunkey-node-selected");
        _this.group.selectAll("rect").classed("sunkey-node-selected", false);
        _this.group.selectAll("path").classed("sunkey-link-selected", false);
        
        if (!selected) 
        {
          d3.select(event.target).classed("sunkey-node-selected", true);

          if(target.type === "token")
          {
            _this.group.selectAll("path").each(function(obj, i, array)
            {
              d3.select(array[i]).classed("sunkey-link-selected", obj.target.id === target.id);

              if(obj.target.id === target.id)
              {  
                var filtered_sentences = _this.stn2class
                  .filter(function(item) { return item.label === obj.source.id; })
                  .map(function(item){ return item.sentence_id; });

                target.sentences.forEach(function(stn, j)
                {
                  if(filtered_sentences.indexOf(stn) > -1)
                  {
                    sentence_ids.push(stn);
                    position.push(target.position[j]);
                  }
                });
              }              
            });
          }
          else
          {
            _this.clicked_class = [target.id];
            var filtered_sentences = _this.stn2class
              .filter(function(item) { return item.label === target.id; })
              .map(function(item){ return item.sentence_id; });
              
            _this.group.selectAll("path").each(function(obj, i, array)
            {
              d3.select(array[i]).classed("sunkey-link-selected", obj.source.id === target.id);

              if(obj.source.id === target.id)
              {  
                obj.target.sentences.forEach(function(stn, j)
                {
                  if(filtered_sentences.indexOf(stn) > -1)
                  {
                    sentence_ids.push(stn);
                    position.push(obj.target.position[j]);
                  }
                });
              }
            });

            if(sentence_ids.length === 0)
              sentence_ids = filtered_sentences;
          }
        }

        _this.call_back.end(sentence_ids, position);
      })
      .on("mouseover", function (event, target) 
      {
        var msg = "";

        if(target.type === "token")
        { 
          // var count_sentences = _this.selected_items.length === 0 ? target.data.sentences.length :  _this.count(_this.selected_items, target);
          // var count_sentences = 0;
          // msg = _this.token_info.get_table(target, count_sentences);

          // _this.tooltip.show(msg, [event.clientX, event.clientY]);
        }  
      })
      .on("mouseout", function (event, target) { _this.tooltip.hide(); });      

    node
      .append("text")
      .attr("x", function(d) { return d.x0 - _this.margin.text_offset; })
      .attr("y", function(d) { return (d.y1 + d.y0) / 2; })
      .attr("text-anchor", "end")
      .text(function(d) { return d.id; })
      .filter(function(d) { return d.x0 < (+_this.svg.attr("width") - _this.margin.left - _this.margin.right) / 2; })
      .attr("x", function(d) { return d.x1 + _this.margin.text_offset; })
      .attr("text-anchor", "start");                
  }
  draw(data, data_summary, palette) 
  {
    var token_selected = false;
    this.svg.select("g").selectAll("rect").each(function (obj, i, dom_obj_list) 
    { 
      token_selected = token_selected || (obj.type == "token" && d3.select(dom_obj_list[i]).classed("sunkey-node-selected"));
    });

    this.svg = this.config_svg();
    this.group = this.svg.select("g")
      .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");            
    this.stn2class = data.sentences;  

    var sankey = d3.sankey()
      .nodeWidth(20)
      .nodePadding(10)
      .size([+this.svg.attr("width") - this.margin.left - this.margin.right, 
             +this.svg.attr("height") - this.margin.top - this.margin.bottom]);

    //Aligns classes on the left-side even with no link connection with tokens     
    sankey.nodeAlign(function(node, n){ return node.type == "class" || node.sourceLinks.length > 0 ? node.depth : n - 1; });
    var graph = sankey(data);

    this.update_link(graph.links);
    this.update_node(sankey, graph.nodes, palette);

    if(token_selected)
      this.call_back.end([], []);
  }
  count(selected_items, obj)
  {
    if(obj.type == "class")
      return obj.sourceLinks.filter(function(item)
      {
        return item.target.sentences
        .filter(function (sentence_id, k) { return selected_items.indexOf(sentence_id) !== -1; })
        .length > 0;        
      })
      .length;
    else
    return obj.sentences
      .filter(function (sentence_id, k) { return selected_items.indexOf(sentence_id) !== -1; })
      .length > 0;  
  }
  get_node_class(obj)
  {
    var class_name = "";

    if(obj.type === "class" && 
       ((this.selected_classes.length > 0 && this.selected_classes.indexOf(obj.id) !== -1) ||
        (this.clicked_class.length > 0 && this.clicked_class.indexOf(obj.id) !== -1)))
      class_name = "sunkey-node-selected";

    return class_name;
  }
  get_link_class(obj)
  {
    var class_ = "sunkey-link";

    if((this.selected_classes.length > 0 && this.selected_classes.indexOf(obj.source.id) !== -1) ||
       (this.clicked_class.length > 0 && this.clicked_class.indexOf(obj.source.id) !== -1))
      class_ += " sunkey-link sunkey-link-selected"; 

    return class_;
  }
  select(data, redraw = true) 
  {
    this.selected_items = data;

    if(redraw)
    {
      var _this = this;
      this.selected_classes = this.stn2class
        .filter(function(obj){ return _this.selected_items.indexOf(obj.sentence_id) !== -1;  })
        .map(function(obj){ return obj.label; });

      this.svg.select("g").selectAll("rect")
        .attr("class", function (obj) { return _this.get_node_class(obj) });        

      this.svg.select("g").selectAll("path")
        .attr("class", function (obj) { return _this.get_link_class(obj); });
    } 
  }
}
