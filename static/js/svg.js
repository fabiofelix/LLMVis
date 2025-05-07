
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
    let svg = d3.select(this.wrapper).select("svg");

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
  clear()
  {
    throw new Error('You have to implement the method clear!');
  }
}

class TokenInfo
{
  get_table(data, count, show_id = true)
  {
    let html = "";

    if(show_id)
      html  += "<p><span class='font-weight-bold'>Token: </span>" + data.id + "</p>";

    html += "<p><span class='font-weight-bold'>Entries: </span>" + count + "</p>";          

    html += "<div class='table-responsive'>";
    html += "<table class='table table-bordered' id='dataTable' width='100%' cellspacing='0'>";
    html += "<thead><tr><th>word</th><th>POS-tag</th><th>Entity</th></tr></thead>";
    html += "<tbody>";
    let aux_search = [];

    for(let i = 0; i < data.postag.length; i++)
    {
      let search_key = data.postag[i][0] + "," + data.postag[i][1] + "," + (data.named_entity[i] === null ? "" : data.named_entity[i]);
    
      if( aux_search.indexOf(search_key) === -1 )
      {  
        aux_search.push(search_key);
        
        html += "<tr>";

        let text = data.postag[i][0];

        let id     = data.id.trim()
        let index  = text.toLowerCase().indexOf(id.toLowerCase());
        let before = text.slice(0, index);
        let after  = text.slice(index + id.length);
        let token  = text.substring(index, index + id.length);        

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
    this.current_selected_class = [];
    this.current_lasso_selection = [];
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
    const _this = this;
    let label_list = [];
    this.svg = this.config_svg();

    let group = this.svg.select("g").attr("class", "scatter-plot");
    let xScale = d3.scaleLinear([data_summary.min_x, data_summary.max_x], [this.circle_size, this.svg.attr("width") - this.circle_size]);
    let yScale = d3.scaleLinear([data_summary.max_y, data_summary.min_y], [this.circle_size, this.svg.attr("height") - this.circle_size]);
    let circle = group.selectAll("circle")
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
        const ids = lassoBrush.selectedItems()["_groups"][0]
          .map(function(item, i) { return item.__data__.sentence_id; });

        if(_this.current_lasso_selection.length !== ids.length)  
        {  
          _this.current_lasso_selection = ids
          _this.call_back.end(ids, "lasso");
        }  
      });
    this.svg.call(lassoBrush);
    this.draw_legend(label_list, palette);
  }
  draw_legend(label_list, palette) 
  {
    label_list.sort();
    this.config_legend();

    const rect_size = 15;
    const legend_group = this.legend.select("g");   
    const _this = this;

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
        _this.current_selected_class = [];
        let text_ids = [];
        const selected = d3.select(this).classed("scatter-legend-selected");
 
        d3.select(this).classed("scatter-legend-selected", !selected);
        _this.legend.selectAll(".scatter-legend-selected").each(function(class_) { _this.current_selected_class.push(class_);  }) 

        if(_this.current_selected_class.length > 0)
          _this.svg.selectAll(".scatter-circle").each(function(obj)
          {
            if(_this.current_selected_class.indexOf(obj.label) !== -1)
              text_ids.push(obj.sentence_id);
          });

        text_ids = _this.call_back.end(text_ids, "class");
        _this.select(text_ids);
      });
  }
  get_legend_class(obj)
  {
    let class_name = "scatter-legend";

    if(this.current_selected_class.length > 0 && this.current_selected_class.indexOf(obj) !== -1)
      class_name += " scatter-legend-selected";

    return class_name;
  }
  get_circle_class(obj)
  {
    let class_name = "scatter-circle";

    if ((this.selected_items.length > 0 && this.selected_items.indexOf(obj.sentence_id) === -1) ||
        (this.current_selected_class.length > 0 && this.current_selected_class.indexOf(obj.label) === -1))
      class_name += " scatter-unselected";

    return class_name;    
  }
  select(data, redraw = true) 
  {
    const _this = this;
    this.selected_items = data;

    if(redraw)
    {
      this.current_selected_class = [];
      this.legend.selectAll(".scatter-legend-selected").each(function(class_) { _this.current_selected_class.push(class_);  })

      this.svg.select("g").selectAll("circle")
        .attr("class", function (obj) { return _this.get_circle_class(obj); });    
    }  
  }
}

class WordCloud extends MySVG
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.token_info = new TokenInfo();
    this.current_selected_word = null;
  }  
  draw(data, data_summary, palette) 
  {
    const _this = this;
    this.svg = this.config_svg();
    const group = this.svg.select("g");
    const size_scale = d3.scaleLinear().domain([data_summary.min_freq, data_summary.max_freq]).range([20, 40]);
    const color_scale = d3.scaleSequential(d3.interpolate("#c8eac8", "#379337")).domain([data_summary.min_freq, data_summary.max_freq]);        

    const layout = d3.layout.cloud()
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
        .attr("class", function (obj) { return _this.get_word_class(obj); })
        .attr("text-anchor", "middle")
        .attr("transform", function(d) { return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")"; })
        .text(function(d) { return d.id; })
        .on("click", function(event, target)
        {
          const selected = d3.select(event.target).classed("word-selected");
          let sentences = [];
          let position = [];
          group.selectAll("text").classed("word-selected", false);
          _this.current_selected_word = null;

          if(!selected)  
          {
            d3.select(event.target).classed("word-selected", true);
            sentences = target.sentences;
            position = target.position;
            _this.current_selected_word = target.id;
          }  

          _this.call_back.end(sentences, position);
        })
        .on("mouseover", function(event, target)
        {
          const msg = _this.token_info.get_table(target, target.frequency, false);
          _this.tooltip.show(msg, [event.clientX, event.clientY]);
        })                
        .on("mouseout", function(event, target) { _this.tooltip.hide(); });
    });
    layout.start();    
  }
  get_word_class(obj)
  {
    let class_name = "word";

    if (this.current_selected_word === obj.id)
      class_name += " word-selected";

    return class_name;    
  }  
  clear()
  {
    const _this = this;
    this.svg.selectAll(".word-selected").each(function(text, i, array) 
    { 
      d3.select(array[i]).classed("word-selected", false);
      _this.current_selected_word = null;
    });
    this.call_back.end([], []); 
  }
  select(data, redraw = true) 
  {

  }    
}

class SankyDiagram extends MySVG 
{
  constructor(wrapper, tooltip) 
  {
    super(wrapper, tooltip);
    this.group = null;
    this.token_info = new TokenInfo();
    this.selected_classes = [];
    this.current_selected_class = [];
    this.current_selected_token = null;
    this._total_links = 5;
    this.token_node_color = "#91b691";
    this.margin = {top: 5, bottom: 5, left: 5, right: 5, text_offset: 6};
  }
  get total_links()
  {
    return this._total_links;
  }
  update_link(links)
  {
    const _this = this;
    const lines = this.group
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
        const msg = "<p><span class='font-weight-bold'>Mean abs. score: </span>" + target.value.toFixed(5) + "</p>";
       _this.tooltip.show(msg, [event.clientX, event.clientY]);

      })
      .on("mouseout", function (event, target) { _this.tooltip.hide(); });            
  }
  update_node(sankey, nodes, palette)
  {
    const _this = this;
    const node = this.group
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
      .on("click", function (event, rect_obj_info) 
      {
        let sentence_ids = [];
        let position = [];
        _this.current_selected_class = [];
        _this.current_selected_token = null;
        
        const selected = d3.select(event.target).classed("sunkey-node-selected");
        _this.group.selectAll("rect").classed("sunkey-node-selected", false);
        _this.group.selectAll("path").classed("sunkey-link-selected", false);

        if (!selected)         
        {
          d3.select(event.target).classed("sunkey-node-selected", true);

          if(rect_obj_info.type === "token")
          {
            _this.current_selected_token = rect_obj_info.id;

            _this.group.selectAll("path").each(function(path, i, array)
            {
              d3.select(array[i]).classed("sunkey-link-selected", path.target.id === rect_obj_info.id);

              if(path.target.id === rect_obj_info.id)
              {
                rect_obj_info.label.forEach(function(label, i)
                {  
                  if(label === path.source.id)
                  { 
                    sentence_ids.push(rect_obj_info.sentences[i])
                    position.push(rect_obj_info.position[i]); 
                  }  
                }); 
              }
            });
          }
          else
          {
            _this.current_selected_class = [rect_obj_info.id];

            _this.group.selectAll("path").each(function(path, i, array)
            {
              d3.select(array[i]).classed("sunkey-link-selected", path.source.id === rect_obj_info.id);

              if(path.source.id === rect_obj_info.id)
              {
                path.target.label.forEach(function(label, i)
                {  
                  if(label === rect_obj_info.id)
                  { 
                    sentence_ids.push(path.target.sentences[i])
                    position.push(path.target.position[i]); 
                  }  
                });
              }              
            });            
          }
        } 

        _this.call_back.end(sentence_ids, position);
      });

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
    let token_selected = false;
    this.svg.select("g").selectAll("rect").each(function (obj, i, dom_obj_list) 
    { 
      token_selected = token_selected || (obj.type == "token" && d3.select(dom_obj_list[i]).classed("sunkey-node-selected"));
    });

    this.svg = this.config_svg();
    this.group = this.svg.select("g")
      .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");            

    const sankey = d3.sankey()
      .nodeWidth(20)
      .nodePadding(5)
      .size([+this.svg.attr("width") - this.margin.left - this.margin.right, 
             +this.svg.attr("height") - this.margin.top - this.margin.bottom]);

    //Aligns classes on the left-side even with no link connection with tokens     
    sankey.nodeAlign(function(node, n){ return node.type == "class" || node.sourceLinks.length > 0 ? node.depth : n - 1; });
    const graph = sankey(data);

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
    let class_name = "";

    if((obj.type === "class" && 
       ((this.selected_classes.length > 0 && this.selected_classes.indexOf(obj.id) !== -1) ||
        (this.current_selected_class.length > 0 && this.current_selected_class.indexOf(obj.id) !== -1))) ||
       (obj.type == "token" &&
        this.current_selected_token === obj.id)) 
      class_name = "sunkey-node-selected";

    return class_name;
  }
  get_link_class(obj)
  {
    let class_name = "sunkey-link";

    if((this.selected_classes.length > 0 && this.selected_classes.indexOf(obj.source.id) !== -1) ||
       (this.current_selected_class.length > 0 && this.current_selected_class.indexOf(obj.source.id) !== -1) ||
       (this.current_selected_token === obj.target.id))
      class_name += " sunkey-link sunkey-link-selected"; 

    return class_name;
  }
}

