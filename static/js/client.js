let FILTER = null;
let MODEL_LIST = null;
let LOADER_CONTROL = null;
let TOOLTIP = null;
let LABEL_COLOR_PALETTE = null;

let PROJECTION_VIEW = null;
let CLUSTER_VIEW = null;
let TEXT_VIEW = null;
let WORD_VIEW = null;

// https://www.geeksforgeeks.org/how-to-execute-after-page-load-in-javascript/
document.addEventListener("DOMContentLoaded", function()
{
  TOOLTIP = new Tooltip("page-top");
  MODEL_LIST = new Model("model_list", "data_model_title");
  LOADER_CONTROL = new LoaderControl("loader_msg", ["wrapper"]);
  FILTER = new FilterManager();

  PROJECTION_VIEW = new Projection("projection_list", "projection_header", "projection_chart_area");
  WORD_VIEW = new WordView("distance_list", "distance_header", "distance_chart_area");
  CLUSTER_VIEW = new Cluster("token_list", "token_header", "token_chart_area");
  TEXT_VIEW = new TextView("text_list", "text_header", "text_area");

  fetch("/config")
  .then(response => response.json())
  .then(data => {
      MODEL_LIST.create_list(data.models);
  });
});

class Tooltip
{
  constructor(wrapper)
  {
    this.div = document.getElementById("tooltip");
    this.msg = null;

    if(this.div == null)
    {
      this.div = document.createElement("div");      
      this.div.id = "tooltip";
      this.div.className = "hidden";

      var aux = document.createElement("p");
      this.msg = document.createElement("span");
      this.msg.id = "value";

      aux.appendChild(this.msg);
      this.div.appendChild(aux);

      document.getElementById(wrapper).appendChild(this.div);
    }
  }
  show(text, position)
  {
    this.msg.innerHTML = text;

    var x_offset = 5, y_offset = 5,
      x = position[0] + x_offset,
      y = position[1] + y_offset;

    if(x + this.div.clientWidth >= this.div.clientWidth)    
      x = position[0] - x_offset - this.div.clientWidth;
    if(y + this.div.clientHeight >= this.div.clientHeight)    
      y = position[1] - y_offset - this.div.clientHeight;        
    else if(y + this.div.clientHeight < 0)
      y = position[1];

    this.div.style.left = x + "px";
    this.div.style.top = y + "px";

    this.div.classList.remove("hidden");   
  }
  hide()
  {
    this.div.classList.add("hidden");
  }
}

class LoaderControl
{
  constructor(div_id, control_list)
  {
    this.div = document.getElementById(div_id);
    this.control_list = control_list;
  }  
  begin()
  {
    this.div.classList.remove("invisible");

    for(var i = 0; i < this.control_list.length; i++)
    {
      var by_class = document.getElementsByClassName(this.control_list[i]);

      for(var j = 0; j < by_class.length; j++)
      {
        by_class[j].classList.add("content-disabled");
      }

      var by_id = document.getElementById(this.control_list[i])

      if(by_id != null)
        by_id.classList.add("content-disabled");
    }
  }
  end()
  {
    this.div.classList.add("invisible");

    for(var i = 0; i < this.control_list.length; i++)
    {
      var by_class = document.getElementsByClassName(this.control_list[i]);

      for(var j = 0; j < by_class.length; j++)
      {
        by_class[j].classList.remove("content-disabled");
      }

      var by_id = document.getElementById(this.control_list[i])

      if(by_id != null)
        by_id.classList.remove("content-disabled");
    }
  }
}

class FilterManager
{
  constructor()
  {
    this.clear_server();
    this.clear_window();
    this.call_back = { onset: [] };
  }
  addEventListener(type, call_back)
  {
    if (!arguments.length)
      return this.call_back;
    if (arguments.length === 1)
      return this.call_back[type];
    if (Object.keys(this.call_back).indexOf(type) > -1)
      this.call_back[type].push(call_back);

    return this;
  }
  clear_server()
  {
    this.server_model = null;
    this.server_projection = null;
    this.server_distance = null;
    this.server_cluster = null;
  }
  clear_window()
  {
    //view_lasso and view_class filter are from projection view
    this.view_lasso = [];
    this.view_class = [];
    this.view_distance = [];
    this.view_word = [];
    this.view_cluster = [];    
    this.view_text = [];
  }
  set_server(filter_type, value)
  {
    if(this.hasOwnProperty("server_" + filter_type))
      this["server_" + filter_type] = value;
    else 
      throw new Error("Filter has no property [server_" + filter_type + "]");  

    return this;
  }
  get_server(filter_type)
  {
    if(filter_type !== undefined)
    {
      if(this.hasOwnProperty("server_" + filter_type))
        return this["server_" + filter_type]

      return null;
    }  

    return {"model": this.server_model, "projection":  this.server_projection, "distance": this.server_distance, "cluster": this.server_cluster};
  }
  set_view(filter_type, value)
  {
    if(this.hasOwnProperty("view_" + filter_type))
    {
      this["view_" + filter_type] = value;

      for(var i = 0; i < this.call_back.onset.length; i++)
        this.call_back.onset[i](filter_type, value);
    }  
    else 
      throw new Error("Filter has no property [view_" + filter_type + "]");  

    return this;      
  }
  get_view(filter_type)  
  {
    if(filter_type == "projection")
      return this.view_lasso.concat(this.view_class);
    else if(filter_type !== undefined)
    {
      if(this.hasOwnProperty("view_" + filter_type))
        return this["view_" + filter_type]

      return null;
    }  

    //================== RETURNS INTERSECTION ==================//
    var keys = Object.keys(this).filter(function(value) {  return value.includes("view_")  });
    var aux_ids = this[keys[0]];

    for(var k = 1; k < keys.length; k++)
    {
      var current_filter = this[keys[k]];

      if(aux_ids.length == 0)
        aux_ids = current_filter;
      else if(current_filter.length > 0)
        aux_ids = aux_ids.filter(function(value) { return current_filter.indexOf(value) !== -1; });
    }

    return aux_ids;    
  }
  count(except)
  {
    var keys = Object.keys(this).filter(function(value) {  return value.includes("view_")  });
    var count = 0;

    for(var k = 0; k < keys.length; k++)
    {
      if(except === undefined || except === null)
        count += this[keys[k]].length;
      else if(except == "projection")
      {
        if(!keys[k].includes("lasso") && !keys[k].includes("class"))
          count += this[keys[k]].length;
      }
      else if(!keys[k].includes(except))
        count += this[keys[k]].length;
    }

    return count;
  }
}

class VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    this.div_list = document.getElementById(div_list_id);
    this.header = document.getElementById(header_id);
    this.div_chart = document.getElementById(chart_id);
    this.filter_type = null;
    this.source_type = null;
    this.label = null;
    this.topic = null;
    this.drawer = null;
  }
  create_list(items)
  {
    var _this = this;
    this.div_list.innerHTML = "";

    for(var i = 0; i < items.length; i++) 
    {
      var link = document.createElement("a");

      link.className = "dropdown-item";
      link.href = "#";
      link.innerHTML = items[i];
      link.addEventListener("click", function(event){ return _this.filter(event); });

      this.div_list.appendChild(link);
    }
  }  
  set_header(title)
  {
    this.header.innerHTML = title;
  }
  extract_data(objs)
  {
    for(var i = 0; i < objs.length; i++) 
    { 
      if(objs[i].type == this.filter_type)
        return objs[i];
    }    

    return null;
  }  
  filter(event)
  {
    event.preventDefault();
    FILTER.set_server(this.filter_type, event.target.innerHTML);
    FILTER.clear_window();

    const URL = new Request("filter", 
      {
        method: "POST",
        body: JSON.stringify({type: this.filter_type, source: this.source_type, config: FILTER.get_server()})
      });    
      
    LOADER_CONTROL.begin();

    fetch(URL)
    .then(response => response.json())
    .then(data => {
      this.show(data);
      LOADER_CONTROL.end();
    });  

  }  
  show(data, clean=true)
  { 
    throw new Error('You have to implement the method show!');
  }
  drawer_callback(data)
  {
    throw new Error('You have to implement the method drawer_callback!');
  }
  select_items(items, redraw = true)    
  {
    this.drawer.select(FILTER.count(this.filter_type) === 0 ? [] : items, redraw);
  }  
}

class Model extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "model";
  }
  show(data)
  {
    this.set_header("LLM embedding visualization - " + data.models[0] );
    
    PROJECTION_VIEW.create_list(data.projections);
    CLUSTER_VIEW.create_list(data.clusters);
    
    //Projection have to come before the others because of the LABEL_COLOR_PALETTE creation
    PROJECTION_VIEW.show(data);
    WORD_VIEW.show(data);
    CLUSTER_VIEW.show(data);
    TEXT_VIEW.show(data);
  }  
}

class Projection extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "projection";
    this.secondary_filter_type = ["lasso", "class"];
    this.source_type = "sentence";
    this.drawer = new ScatterPlot(chart_id, TOOLTIP);
    var _this = this;
    this.drawer.on("end", function(data, second_filter_type){ return _this.drawer_callback(data, second_filter_type); });
    this.label = null;
    this.topic = null;
  }
  drawer_callback(data, second_filter_type)
  {
    var text_ids = data
      .filter(function(value, index, array) { return array.indexOf(value) === index; });
    text_ids = FILTER.set_view(second_filter_type, text_ids).get_view();

    WORD_VIEW.select_items(text_ids);
    CLUSTER_VIEW.select_items(text_ids);
    TEXT_VIEW.select_items(text_ids);

    return text_ids;
  }
  show(data, clean=true)
  {
    var objs = this.extract_data(data.objs);
    this.set_header("Text - " + objs.data.length + " samples - " + objs.name + " - sh: " + objs.silhouette.toFixed(4) );
    var sum = {min_x: Number.MAX_VALUE, min_y: Number.MAX_VALUE, max_x: Number.MIN_VALUE, max_y: Number.MIN_VALUE};
    var unique_label = [];

    if(objs.label !== null)
      this.label = objs.label;
    if(objs.topic !== null)
      this.topic = objs.topic;   

    var _this = this; 

    var formated_objs = objs.data.map(function(value, idx)
    {
      sum.min_x = Math.min(sum.min_x, value[0]);
      sum.max_x = Math.max(sum.max_x, value[0]);
      
      sum.min_y = Math.min(sum.min_y, value[1]);
      sum.max_y = Math.max(sum.max_y, value[1]);

      var label = _this.label[idx] == null ? _this.topic[idx] : _this.label[idx];

      if(unique_label.indexOf(label) == -1)
        unique_label.push(label);

      return {sentence_id: objs.ids[idx], x: value[0], y: value[1], label: label};
    });

    LABEL_COLOR_PALETTE = d3.scaleOrdinal(d3.schemeCategory10).domain(unique_label);    
    this.drawer.draw(formated_objs, sum, LABEL_COLOR_PALETTE);
  }
}

class Distance extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "distance";
    this.source_type = "sentence";
    this.drawer = new Heatmap(chart_id);
    var _this = this;
    this.drawer.on("end", function(data){ _this.drawer_callback(data); });
  }
  drawer_callback(data)
  {
    var text_ids = data
      .filter(function(value, index, array) { return array.indexOf(value) === index; });
    text_ids = FILTER.set_view(this.filter_type, text_ids).get_view();
    
    PROJECTION_VIEW.select_items(text_ids);
    CLUSTER_VIEW.select_items(text_ids);
    TEXT_VIEW.select_items(text_ids);
  }  
  show(data, clean=true)
  {
    var objs = this.extract_data(data.objs);
    this.set_header("Text - " + objs.name + " (high-dimension)" + " - sh: " + objs.silhouette.toFixed(4));
    var sum = {min_value: Number.MAX_VALUE, max_value: Number.MIN_VALUE, n_row: objs.data.length, n_col: objs.data.length};

    var formated_objs = []

    for(var i = 0; i < objs.data.length; i++)      
    {
      var aux = [];

      for(var j = 0; j < objs.data.length; j++)      
      {
        sum.min_value = Math.min(sum.min_value, objs.data[i][j]);
        sum.max_value = Math.max(sum.max_value, objs.data[i][j]);      

        aux.push({id_i: objs.ids[i], id_j: objs.ids[j], value: objs.data[i][j]});
      }  

      formated_objs.push(aux);
    }

    this.drawer.draw(formated_objs, sum, null);
  }
}

class WordView extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "word";
    this.source_type = "token";    
    this.drawer = new WordCloud(chart_id, TOOLTIP);
    this.words = null;
    var _this = this;
    this.drawer.on("end", function(data, position){ _this.drawer_callback(data, position); });
    FILTER.addEventListener("onset", function(filter_type, value)
    {
      if(filter_type !== _this.filter_type && value.length === 0)
      {
        var text_id = FILTER.set_view(_this.filter_type, []).get_view();

        PROJECTION_VIEW.select_items(text_id, PROJECTION_VIEW.secondary_filter_type.indexOf(filter_type) !== -1);
        CLUSTER_VIEW.select_items(text_id, CLUSTER_VIEW.filter_type === filter_type);
        TEXT_VIEW.select_items(text_id, TEXT_VIEW.filter_type === filter_type);
      }  
    });
  }  
  drawer_callback(data, position)
  {
    var data_filtered = FILTER.set_view(this.filter_type, data).get_view();
    var text_ids = {};

    for(var i = 0; i < data.length; i++)
    {
      if(data_filtered.indexOf(data[i]) !== -1)
      {
        if(Object.keys(text_ids).indexOf(data[i]) == -1) 
          text_ids[data[i]] = [];

        text_ids[data[i]].push( position[i] );
      } 
    }    
    
    PROJECTION_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    CLUSTER_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    TEXT_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : text_ids);
  }  
  zipf_law(words)
  {
    var samples = 50;
    var min_freq = 0.05;
    var max_freq = 0.95;
    var max_norm_factor = words[words.length - 1].frequency;
    var filtered = words;
    
    if(words[0].frequency !== words[words.length - 1].frequency)
      filtered = words.filter(function(item) { return item.frequency / max_norm_factor > min_freq && item.frequency / max_norm_factor < max_freq; });

    return filtered.length > samples ? filtered.slice(-samples) : filtered;    
  }    
  show(data, clean=true)
  {
    var objs = this.extract_data(data.objs);
    this.words = objs.ids
    .map(function(value, index)
    {
      return {
        id: value, 
        text: value,
        frequency: objs.data.sentences[index].length, 
        sentences: objs.data.sentences[index],
        position: objs.data.position[index], 
        named_entity: objs.data.named_entity[index], 
        postag: objs.data.postag[index],
        word: objs.data.word[index],      
      };
    })
    .sort(function(a, b) { return a.frequency - b.frequency; })
    
    var filtered = this.zipf_law(this.words);
    this.set_header("Token - " + filtered.length + " samples more frequent");
    
    this.drawer.draw(filtered, {min_freq: filtered[0].frequency, max_freq: filtered[filtered.length - 1].frequency});
  }
  select_items(items, redraw = true)
  {
    var filtered = this.words;

    if (FILTER.count(this.filter_type) != 0)
    {
      filtered = this.words
       .map(function(item) {  return Object.assign({}, item); })
       .filter(function(value, index, array)
       { 
          value.frequency = value.sentences
            .filter(function(stn) {  return items.indexOf(stn) !== -1;  })
            .length;
          return value.frequency > 0;
       })
       .sort(function(a, b) { return a.frequency - b.frequency; });
    }

    filtered = this.zipf_law(filtered);
    this.set_header("Token - " + filtered.length + " samples more frequent");

    if(filtered.length === 0)
      this.drawer.draw(filtered, {min_freq: 0, max_freq: 0});
    else
      this.drawer.draw(filtered, {min_freq: filtered[0].frequency, max_freq: filtered[filtered.length - 1].frequency});
  }    
}

class Cluster extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "cluster";
    this.source_type = "token";
    this.drawer = new TreeMap(chart_id, TOOLTIP);    
    var _this = this;
    this.drawer.on("end", function(data, position){ _this.drawer_callback(data, position); });
  }
  drawer_callback(data, position)
  {
    var data_filtered = data
      .filter(function(value, index, array) { return array.indexOf(value) === index; });
    data_filtered = FILTER.set_view(this.filter_type, data_filtered).get_view();
    var text_ids = {};

    for(var i = 0; i < data.length; i++)
    {
      if(data_filtered.indexOf(data[i]) !== -1)
      {
        if(Object.keys(text_ids).indexOf(data[i]) == -1) 
          text_ids[data[i]] = [];

        text_ids[data[i]].push( position[i] );
      } 
    }    
    
    PROJECTION_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    WORD_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    TEXT_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : text_ids);
  }  
  show(data, clean=true)
  {
    var objs = this.extract_data(data.objs);
    this.set_header("Token - " + objs.ids.length + " samples - " + objs.name);

    let unique_cluster = [...new Set(objs.data.clusters)]; 
    let tree = {id: "root", children: []};
    unique_cluster.sort(function(a, b) { return a - b; });

    for(var i = 0; i < unique_cluster.length; i++)
    {
      var token_indices = objs.data.clusters
        .map(function(value, index){ return value === unique_cluster[i] ? index : -1 }) // position index of cluster == unique_cluster[i]
        .filter(function(value){ return value !== -1 }); // only valide positions != -1
      var cluster = {id: "cluster" + unique_cluster[i], main_token: objs.data.main_token[i], children: []};

      token_indices.forEach(function(index)
      {
        cluster.children.push({ 
          id: objs.ids[index], 
          sentences: objs.data.sentences[index],  
          position: objs.data.position[index], 
          named_entity: objs.data.named_entity[index], 
          postag: objs.data.postag[index],
          word: objs.data.word[index],
         });
      });

      tree.children.push(cluster);      
    }

    this.drawer.draw(tree);
  }
}

class TextView extends VisManager
{
  constructor(div_list_id, header_id, chart_id)
  {
    super(div_list_id, header_id, chart_id);
    this.filter_type = "text";
    this.objs = null;
    this.highlight = new HighlightText();    
    this.paginator = new PaginateText("text-previous", "text-next", "text-first", "text-last", "text-position", chart_id); 

    var _this = this;

    this.paginator.on("paginate", function(obj, text_tokens)
    {
      _this.highlight.execute(obj.dataset.text, _this.get_unique_token(text_tokens), obj.querySelectorAll("span")[1]);
    });

    document.getElementById("clear_text_selection").addEventListener("click", function(event){ _this.clear_all(event) }); 
  }
  show(data, clean=true)
  { 
    var _this = this;
    this.objs  = this.extract_data(data.objs);    
    this.label = this.objs.data.label[0] == null ? this.objs.data.topic : this.objs.data.label;
    var row = document.createElement("div");
    row.className = "row overflow-auto";

    this.div_chart.innerHTML = "";
    this.div_chart.appendChild(row);

    this.paginator.count_text = this.objs.data.text.length;

    for (var idx = 0; idx < this.objs.data.text.length; idx++)
    {
      var col = document.createElement("div");
      col.className = "col-xl-12 col-md-6 mb-4"
      col.dataset.id = this.objs.ids[idx];
      col.dataset.text = this.objs.data.text[idx];

      var card = document.createElement("div");
      card.className = "card border-left-primary h-100 py-2";
      card.style.cssText = "border-left-color: " + LABEL_COLOR_PALETTE(this.label[idx]) + " !important";

      var body = document.createElement("div");
      body.className = "card-body";

      var title = document.createElement("span");
      title.className = "font-weight-bold text-id";
      title.innerHTML = this.objs.ids[idx] + " (" + this.label[idx] + "):";
      title.data = this.objs.ids[idx];

      var text = document.createElement("span");
      text.innerHTML = this.objs.data.text[idx];   
      
      body.appendChild(title)
      body.appendChild(text);
      card.appendChild(body);
      col.appendChild(card);
      row.appendChild(col);

      title.addEventListener("click", function(event) { _this.drawer_callback(event); }); 

      this.paginator.text_hidden(idx, col);
    }

    this.paginator.manage_control();
  }
  clear_all(event)
  {
    event.preventDefault();
    var _this = this;
    document.querySelectorAll(".text-selected").forEach(function(element)
    {
      element.classList.remove("text-selected");   
      var text_id = FILTER.set_view(_this.filter_type, []).get_view();

      PROJECTION_VIEW.select_items(text_id);
      WORD_VIEW.select_items(text_id);
      CLUSTER_VIEW.select_items(text_id);    
    });    
  }
  drawer_callback(event)
  {
    event.preventDefault();

    var text_id = FILTER.get_view(this.filter_type);

    if(event.target.classList.contains("text-selected"))
    {
      event.target.classList.remove("text-selected");   
      var index = text_id.indexOf(event.target.data);
      text_id.splice(index, 1);
    }  
    else 
    {
      event.target.classList.add("text-selected");
      text_id.push(event.target.data);
    }  

    text_id = FILTER.set_view(this.filter_type, text_id).get_view();

    PROJECTION_VIEW.select_items(text_id);
    WORD_VIEW.select_items(text_id);
    CLUSTER_VIEW.select_items(text_id);    
  }
  get_unique_token(text_tokens)
  {
    var aux = [];

    for(var jdx = 0; jdx < text_tokens.length; jdx++)
    {
      var found_index = -1;

      for(var kdx = 0; kdx < aux.length; kdx++)
      {
        if(aux[kdx][0] === text_tokens[jdx][0] && aux[kdx][1][0] === text_tokens[jdx][1][0] && aux[kdx][1][1] === text_tokens[jdx][1][1])
        {
          found_index = kdx;
          break;
        }
      }

      if(found_index === -1)
        aux.push(text_tokens[jdx]);
    }

    return aux;
  }
  select_items(items)
  {
    this.paginator.set_selected_text(FILTER.count(this.filter_type) === 0 ? [] : items);
    this.paginator.paginate_text();
    this.paginator.manage_control();
  }      
}

class HighlightText
{
  execute(original_text, text_tokens, html_tag)
  { 
    //text_tokens: list of [token, [start pos, end pos]]
    text_tokens.sort(function(a, b) { return a[1][0] - b[1][0]; });
    var unique_token_id = text_tokens
      .map(function(obj){ return obj[0]; })
      .filter(function(value, index, array){ return array.indexOf(value) === index; })
      .sort();    
    var palette = d3.scaleOrdinal(d3.schemeSet3).domain(unique_token_id);
    var new_text = original_text;

    for(var jdx = text_tokens.length - 1; jdx >= 0; jdx--)  
    {
      var before = new_text.slice(0, text_tokens[jdx][1][0]);
      var after  = new_text.slice(text_tokens[jdx][1][1]);
      var token  = new_text.substring(text_tokens[jdx][1][0], text_tokens[jdx][1][1]);

      new_text = before + "<span style='background-color:" + palette(text_tokens[jdx][0]) + "'>" + token + "</span>" + after;
    }

    html_tag.innerHTML = new_text;    
  }
}

class PaginateText
{
  constructor(prev_id, next_id, first_id, last_id, pos_id, text_wrapper)
  {
    this.text_wrapper = document.getElementById(text_wrapper);
    this.call_back = { paginate: function () { } };

    this.first_page = document.getElementById(first_id);
    this.first_page.classList.add("disabled");

    this.previous_page = document.getElementById(prev_id);
    this.previous_page.classList.add("disabled");
    
    this.next_page = document.getElementById(next_id);
    this.next_page.classList.add("disabled");

    this.last_page = document.getElementById(last_id);
    this.last_page.classList.add("disabled");
        
    this.page_position = document.getElementById(pos_id);

    this.inc = 10;
    this.first_idx = 0;
    this.count_text = 0;
    this.selected_text = [];
    this.count_selected_text = 0; 

    this.set_event();
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
  get_count_text()
  {
    return this.count_selected_text > 0 ? this.count_selected_text : this.count_text;
  }   
  set_event()
  {
    var _this = this;

    this.first_page.addEventListener("click", function(event)
    {
      event.preventDefault();
      _this.first_idx = 0;

      _this.manage_control();
      _this.paginate_text();
    });    
    this.previous_page.addEventListener("click", function(event)
    {
      event.preventDefault();
      _this.first_idx -= _this.inc;

      _this.manage_control();
      _this.paginate_text();
    });
    this.next_page.addEventListener("click", function(event)
    {
      event.preventDefault();
      _this.first_idx += _this.inc;

      _this.manage_control();
      _this.paginate_text();
    });
    this.last_page.addEventListener("click", function(event)
    {
      event.preventDefault();
      var aux_idx = Math.floor(_this.get_count_text() / _this.inc) * _this.inc;
      _this.first_idx = aux_idx == _this.get_count_text() ? _this.get_count_text() - _this.inc : aux_idx;

      _this.manage_control();
      _this.paginate_text();
    });     
  }
  set_selected_text(items)
  {
    this.selected_text = items;
    this.first_idx = 0;
  }
  paginate_text()
  {
    let cols = this.text_wrapper.querySelectorAll(".col-xl-12");
    let ids = Object.keys(this.selected_text);
    this.count_selected_text = 0;

    for(var idx = 0; idx < cols.length; idx++)
    {
      this.call_back.paginate(cols[idx], []);
      var hidden = false;

      //Selection from: scatterplot and heatmap
      if(this.selected_text instanceof Array)
      {
        hidden = this.selected_text.length > 0 && this.selected_text.indexOf(cols[idx].dataset.id) == -1;
      }
      //Selection from: treemap
      else if(ids.length > 0)
      {
        hidden = ids.indexOf(cols[idx].dataset.id) == -1;

        if(!hidden)
        {
          var text_tokens = this.selected_text[cols[idx].dataset.id];
          this.call_back.paginate(cols[idx], text_tokens);
        }
      }

      if(!hidden)
        ++this.count_selected_text;

      this.text_hidden(this.selected_text.length == 0 ? idx : this.count_selected_text, cols[idx], hidden);
    }     
  }
  text_hidden(idx, obj, hidden = false)
  {
    obj.classList.remove("hidden");

    if(hidden || idx < this.first_idx || idx >= (this.first_idx + this.inc))
      obj.classList.add("hidden"); 
  }
  manage_control()
  {
    this.page_position.innerHTML = (this.first_idx + 1) + "-" + Math.min(this.first_idx + this.inc, this.get_count_text());

    if(this.first_idx == 0)
    {
      this.first_page.classList.add("disabled");
      this.previous_page.classList.add("disabled");
    }
    else
    {
      this.first_page.classList.remove("disabled");
      this.previous_page.classList.remove("disabled");
    }     

    if(this.first_idx + this.inc >= this.get_count_text())
    {  
      this.next_page.classList.add("disabled");
      this.last_page.classList.add("disabled");
    }     
    else
    {
      this.next_page.classList.remove("disabled");      
      this.last_page.classList.remove("disabled");      
    }
  }         
}
