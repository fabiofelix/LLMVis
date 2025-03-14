// https://www.geeksforgeeks.org/binary-search-in-javascript/
function binary_search(arr, x, start, end) 
{
  if(start === undefined)
    start = 0;
  if(end === undefined)
    end = arr.length - 1;

  // Base Condition
  if(start > end) 
    return -1;

  // Find the middle index
  let mid = Math.floor((start + end) / 2);

  // Compare mid with given key x
  if (arr[mid] === x) 
    return mid;

  // If element at mid is greater than x,
  // search in the left half of mid
  if (arr[mid] > x)
    return binary_search(arr, x, start, mid - 1);
  else
    // If element at mid is smaller than x,
    // search in the right half of mid
    return binary_search(arr, x, mid + 1, end);
}

//  https://nileshsaini09.medium.com/searching-algorithms-in-javascript-8954272914cc
function interpolation_search(arr, k)
{
  let n = arr.length;
  let low = 0, high = n - 1;

  while(low <= high && k >= arr[low] && k <= arr[high]) 
  {
    // finding the position 
    let pos = low + 
        Math.floor(((k - arr[low]) * (high - low)) / (arr[high] - arr[low]));
  
    if(arr[pos] < k) 
      low = pos + 1;
    else if(arr[pos] > k) 
     high = pos - 1;
    else 
      return pos;
  }

  return -1;
}


class MyCanvas
{
  constructor(wrapper)
  {
    this.wrapper = document.getElementById(wrapper);
    this.canvas = this.config_canvas();
    this.call_back = {start: function(){}, draw: function(){}, end: function(){}};
  }
  on(type, call_back) 
  {
    if(!arguments.length) 
      return this.call_back;
    if(arguments.length === 1) 
      return this.call_back[type];
    if(Object.keys(this.call_back).indexOf(type) > -1) 
      this.call_back[type] = call_back;

    return this;
  }
  config_canvas()
  {
    var canvas = d3.select(this.wrapper).select("canvas");

    if(canvas.empty())
      canvas = d3.select(this.wrapper).append("canvas");
    else
      canvas.selectAll("*").remove();        

    canvas
      .attr("width", this.wrapper.clientWidth)
      .attr("height", this.wrapper.clientHeight);

    return canvas;  
  };
  draw(data, data_summary, palette)
  {
    throw new Error('You have to implement the method draw!');
  }
  select(data)
  {
    throw new Error('You have to implement the method select!');
  }  
}

class Heatmap extends MyCanvas
{
  constructor(wrapper)
  {
    super(wrapper);
    this.legend = this.config_legend();
    this.data = [];
    this.data_summary = null;
    this.selected_items = [];
    this.axis = {"x": [], "y": []};
    this.init_svg_selection();
  }
  config_canvas_aux(width, height)
  {
    var canvas = d3.select(this.wrapper).select("#canvas_aux");

    if(canvas.empty())
      canvas = d3.select(this.wrapper).append("canvas");
    else
      canvas.selectAll("*").remove();        

    canvas
      .attr("id", "canvas_aux")
      .attr("width", width)
      .attr("height", height);

    return canvas;  
  };  
  config_legend()
  {
    var canvas = d3.select(this.wrapper).append("canvas");
    canvas
      .attr("width", this.wrapper.clientWidth)
      .attr("height", 15);    

    return canvas;  
  }
  init_svg_selection()
  {
    var _this = this;

    this.selection = d3.select(this.wrapper).select("svg")
      .on("click", function(event) { _this.update_svg_selection(event); });

    if(this.selection.empty())
      this.selection = d3.select(this.wrapper).append("svg");
    else
      this.selection.selectAll("*").remove();

    this.selection
      .attr("width", this.wrapper.clientWidth)
      .attr("height", this.wrapper.clientHeight)
      .attr("class", "heatmap-svg")
      .append("g");
  }
  update_svg_selection(event)
  {
    var rect = this.selection.select("rect");

    if(rect.empty())
    {  
      var _this = this;

      let previous_point = [];
      let init_size = 100;

      var point = d3.pointer(event, event.target);
      var x = Math.min(_this.wrapper.clientWidth - init_size, Math.max(0, point[0] - init_size / 2));
      var y = Math.min(_this.wrapper.clientHeight - init_size, Math.max(0, point[1] - init_size / 2));
  
      this.selection.select("g")
        .append("rect")
        .attr("class", "heatmap-select")
        .attr("x", x)
        .attr("y", y)        
        .attr("width", init_size)
        .attr("height", init_size)
        .call(
          d3.drag()
          .on("start", function(event)
          {
            d3.select(this).classed("heatmap-dragging", true);
            previous_point = d3.pointer(event, this);
          })
          .on("drag", function(event)
          {
            let d3_this = d3.select(this);
            let point = d3.pointer(event, this);
            let new_x = +d3_this.attr("x") + point[0] - previous_point[0];
            let new_y = +d3_this.attr("y") + point[1] - previous_point[1];
  
            d3_this
              .attr("x", Math.min(_this.wrapper.clientWidth - +d3_this.attr("width"), Math.max(0, new_x)))
              .attr("y", Math.min(_this.wrapper.clientHeight - +d3_this.attr("height"), Math.max(0, new_y)));
  
            previous_point = point;
          })
          .on("end", function(event)
          {
            d3.select(this).classed("heatmap-dragging", false);
            var rect = d3.select(event.sourceEvent.target);
            var x_range = [+rect.attr("x"), +rect.attr("x") + +rect.attr("width")];
            var y_range = [+rect.attr("y"), +rect.attr("y") + +rect.attr("height")];
            let selected_items = [];

            for(var i = 0; i < _this.data.length; i++)
            {
              if(_this.axis["x"][i].value >= x_range[0] && _this.axis["x"][i].value <= x_range[1])
                 selected_items.push(_this.axis["x"][i].id);
              if(_this.axis["y"][i].value >= y_range[0] && _this.axis["y"][i].value <= y_range[1])
                selected_items.push(_this.axis["y"][i].id);
            }

            _this.call_back.end(selected_items);
          })
        );
    }  
    else
    {
      this.selection.select("g").selectAll("*").remove();  
      this.call_back.end([]);  
    }  
  }  
  draw(data, data_summary, palette)
  {
    if(data !== undefined)
    {
      this.data = data;
      this.data_summary = data_summary;
      this.selected_items = [];
      this.axis = {"x": [], "y": []};
    }      

    let context = this.canvas.node().getContext("2d");
    palette = d3.scaleSequential(d3.interpolateOranges).domain([this.data_summary.min_value, this.data_summary.max_value]);

    let xScale = d3.scaleLinear()
      .domain([0, this.data_summary.n_col])
      .range([0, +this.canvas.attr("width")]);
    let yScale = d3.scaleLinear()
      .domain([0, this.data_summary.n_row])
      .range([+this.canvas.attr("height"), 0]);
    
    this.canvas_aux = this.config_canvas_aux(this.data.length, this.data.length);

    let context_aux = this.canvas_aux.node().getContext("2d");      
    let image = context_aux.createImageData(this.data.length, this.data.length);

    for(var i = 0; i < this.data.length; i++)
    {
      if(data !== undefined)
        this.axis["y"].push({id: this.data[i][0].id_i, value: yScale(i)});

      for(var j = 0; j < this.data.length; j++)
      {
        var color = palette(this.data[i][j].value);
        color = color.replace("rgb(", "").replace(")", "").split(",");
        var alpha = 255;

        if(this.selected_items.length > 0 && binary_search(this.selected_items, this.data[i][j].id_i) === -1 && binary_search(this.selected_items, this.data[i][j].id_j) === -1)   
          alpha = Math.ceil(0.1 * 255);        
        if(data !== undefined && i == 0)
          this.axis["x"].push({id: this.data[i][j].id_j, value: xScale(j)});

        image.data[i * this.data.length * 4 + j * 4 + 0] = +color[0];
        image.data[i * this.data.length * 4 + j * 4 + 1] = +color[1];
        image.data[i * this.data.length * 4 + j * 4 + 2] = +color[2];
        image.data[i * this.data.length * 4 + j * 4 + 3] = alpha;
      }
    }    

    context_aux.putImageData(image, 0, 0);
    context.drawImage(this.canvas_aux.node(), 0, 0, +this.canvas.attr("width"), +this.canvas.attr("height"));
    this.canvas_aux.classed("hidden", true);

    if(data !== undefined)
      this.init_svg_selection();

    this.draw_legend();
  }
  draw_legend()
  {
    let n_col = 512;
    let legend_context = this.legend.node().getContext("2d");
    let legend_width = this.legend.attr("width") / n_col;
    let legend_height = this.legend.attr("height");

    var palette = d3.scaleSequential(d3.interpolateOranges).domain([0, n_col]);
    var xScale = d3.scaleLinear()
      .domain([0, n_col])
      .range([0, +this.legend.attr("width")]);    

    for(let value = 0; value < n_col; value++)  
    {
      legend_context.fillStyle = palette(value);
      legend_context.fillRect(xScale(value), 0, legend_width, legend_height);
    }
    
    legend_context.font = "12px sans-serif";

    for(let value = 0; value < n_col; value++)  
    {
      if(value == 0)
      {  
       legend_context.fillStyle = "#000000";
       legend_context.fillText("Low", xScale(value), 10);
      } 
      else if(value == n_col - 1)
      {
        var text_width = legend_context.measureText("High").width;
        legend_context.fillStyle = "#ffffff";
        legend_context.fillText("High", xScale(value) - text_width, 10);      
      }  
    }
  }
  select(data)
  {
    this.selected_items = data;
    this.selected_items.sort();
    this.canvas = this.config_canvas();
    this.draw();
  }      
}
