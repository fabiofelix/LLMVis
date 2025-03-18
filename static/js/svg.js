
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
  get_table(data, count_sentences, show_id = true)
  {
    var html = "";

    if(show_id)
      html  += "<p><span class='font-weight-bold'>Token: </span>" + data.id + "</p>";

    html += "<p><span class='font-weight-bold'>Sentences: </span>" + count_sentences + "</p>";          

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

        html += "<td>" + text.replace(data.id,  "<span class='word-part'>" + data.id + "</span>") + "</td>";
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
    this.legend = this.config_legend();
    this.circle_size = 3;
    this.selected_class = [];
  }
  config_legend() 
  {
    var svg = d3.select(this.wrapper).append("svg");

    svg
      .attr("width", this.wrapper.clientWidth)
      .attr("height", 20)
      .append("g");

    return svg;
  }
  draw(data, data_summary, palette) 
  {
    var _this = this;
    var label_list = [];
    this.svg = this.config_svg();

    var group = this.svg.select("g");
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
      .style("fill", function (d) { label_list.push(d.label); return palette(d.label); });

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
    label_list = [...new Set(label_list)];
    var legend_group = this.legend.select("g");
    var rect_size = 15;
    var _this = this;

    legend_group.selectAll("rect")
      .data(label_list)
      .enter()
      .append("rect")
      .attr("class", "scatter-legend")
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

