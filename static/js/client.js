let FILTER = null;
let MODEL_LIST = null;
let LOADER_CONTROL = null;
let TOOLTIP = null;
let LABEL_COLOR_PALETTE = null;

let PROJECTION_VIEW = null;
let TEXT_VIEW = null;
let WORD_VIEW = null;
let EXPLANATION_VIEW = null;

// https://www.geeksforgeeks.org/how-to-execute-after-page-load-in-javascript/
document.addEventListener("DOMContentLoaded", function()
{
  TOOLTIP = new Tooltip("page-top");
  MODEL_LIST = new Model("model_list", "data_model_title");
  LOADER_CONTROL = new LoaderControl("loader_msg", ["wrapper"]);
  FILTER = new FilterManager();

  PROJECTION_VIEW = new Projection("projection_list", "projection_header", "projection_chart_area");
  WORD_VIEW = new WordView("wordcloud_list", "wordcloud_header", "wordcloud_chart_area");
  EXPLANATION_VIEW = new Explanation("explain_list", "explain_header", "explain_chart_area", "explain_info")
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

      const aux = document.createElement("p");
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

    let x_offset = 3, y_offset = 3,
        x = position[0] + x_offset,
        y = position[1] + y_offset;

    this.div.classList.remove("hidden");

    if(x + this.div.clientWidth >= window.innerWidth)    
      x = window.innerWidth - this.div.clientWidth;
    if(y + this.div.clientHeight >= window.innerHeight)    
      y = window.innerHeight - this.div.clientHeight;
    else if(y + this.div.clientHeight < 0)
      y = position[1];

    this.div.style.left = x + "px";
    this.div.style.top = y + "px";
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

    for(let control_name of this.control_list)
    {
      let control_by_class = document.getElementsByClassName(control_name);

      for(let control of control_by_class)
      {
        control.classList.add("content-disabled");
      }

      let control_by_id = document.getElementById(control_name)

      if(control_by_id != null)
        control_by_id.classList.add("content-disabled");
    }
  }
  end()
  {
    this.div.classList.add("invisible");

    for(let control_name of this.control_list)
    {
      let control_by_class = document.getElementsByClassName(control_name);

      for(let control of control_by_class)
      {
        control.classList.remove("content-disabled");
      }

      let control_by_id = document.getElementById(control_name)

      if(control_by_id != null)
        control_by_id.classList.remove("content-disabled");
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
    if (Object.keys(this.call_back).includes(type))
      this.call_back[type].push(call_back);

    return this;
  }
  clear_server()
  {
    this.server_model = null;
    this.server_projection = null;
    this.server_distance = null;
    this.server_cluster = null;
    this.server_explanation = null;
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
    this.view_explanation = [];
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

    const key_list = Object.keys(this).filter(function(value) {  return value.includes("server_")  });
    let config = {};

    for(let key of key_list)
    {
      config[ key.split("_")[1] ] = this[key];
    }    

    return config;
  }
  set_view(filter_type, value)
  {
    if(this.hasOwnProperty("view_" + filter_type))
    {
      this["view_" + filter_type] = value;

      for(let i = 0; i < this.call_back.onset.length; i++)
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
    const key_list = Object.keys(this).filter(function(value) {  return value.includes("view_")  });
    let aux_ids = this[key_list[0]];

    for(let k = 1; k < key_list.length; k++)
    {
      let current_filter = this[key_list[k]];

      if(aux_ids.length == 0)
        aux_ids = current_filter;
      else if(current_filter.length > 0)
        aux_ids = aux_ids.filter(function(value) { return current_filter.includes(value); });
    }

    return aux_ids;    
  }
  count(except)
  {
    const key_list = Object.keys(this).filter(function(value) {  return value.includes("view_")  });
    let count = 0;

    for(let key of key_list)
    {
      if(except === undefined || except === null)
        count += this[key].length;
      else if(except == "projection")
      {
        if(!key.includes("lasso") && !key.includes("class"))
          count += this[key].length;
      }
      else if(!key.includes(except))
        count += this[key].length;
    }

    return count;
  }
}

class VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    this.div_list = document.getElementById(div_list_id);
    this.header = document.getElementById(header_id);
    this.div_chart = document.getElementById(chart_id);
    this.info_button = info_id === undefined ? null : document.getElementById(info_id);
    this.filter_type = null;
    this.source_type = null;
    this.drawer = null;

    if(this.info_button !== null)
    {
      const _this = this;

      this.info_button.addEventListener("mouseover", function(event)
      { 
        const info = _this.get_info();

        if(info !== null)
          TOOLTIP.show(info, [event.clientX, event.clientY]); 
      });
      this.info_button.addEventListener("mouseout", function(event){  TOOLTIP.hide(); });
    }  
  }
  create_list(items)
  {
    const _this = this;
    this.div_list.innerHTML = "";

    for(let text of items) 
    {
      let link = document.createElement("a");

      link.className = "dropdown-item";
      link.href = "#";
      link.innerHTML = text;
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
    for(let obj of objs)
    { 
      if(obj.type == this.filter_type)
        return obj;
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
  get_info()
  {
    throw new Error('You have to implement the method get_info!');
  }
  select_items(items, redraw = true)    
  {
    this.drawer.select(FILTER.count(this.filter_type) === 0 ? [] : items, redraw);
  }  
}

class Model extends VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    super(div_list_id, header_id, chart_id, info_id);
    this.filter_type = "model";
  }
  show(data)
  {
    this.set_header("LLM embedding visualization - " + data.models[0] );
    
    PROJECTION_VIEW.create_list(data.projections);
    EXPLANATION_VIEW.create_list(data.explanations);
    
    //Projection have to come before the others because of the LABEL_COLOR_PALETTE creation
    PROJECTION_VIEW.show(data);
    WORD_VIEW.show(data);
    EXPLANATION_VIEW.show(data);
    TEXT_VIEW.show(data);
  }  
}

class Projection extends VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    super(div_list_id, header_id, chart_id, info_id);
    this.filter_type = "projection";
    this.secondary_filter_type = ["lasso", "class"];
    this.source_type = "sentence";
    this.drawer = new ScatterPlot(chart_id, TOOLTIP);
    const _this = this;
    this.drawer.on("end", function(data, second_filter_type){ return _this.drawer_callback(data, second_filter_type); });
    this.sentences_ = null;
  }
  get sentences()
  {
    return this.sentences_;
  }
  drawer_callback(data, second_filter_type)
  {
    const text_ids = FILTER.set_view(second_filter_type, data).get_view();

    WORD_VIEW.select_items(text_ids);
    const text_label = this.sentences_.filter(function(stn){ return text_ids.includes(stn.sentence_id); });
    EXPLANATION_VIEW.select_items(text_label);
    TEXT_VIEW.select_items(text_ids);

    return text_ids;
  }
  show(data, clean=true)
  {
    const objs = this.extract_data(data.objs);
    this.set_header("Text - " + objs.data.length + " samples - " + objs.name + " - sh: " + objs.silhouette.toFixed(4) );
    const sum = {min_x: Number.MAX_VALUE, min_y: Number.MAX_VALUE, max_x: Number.MIN_VALUE, max_y: Number.MIN_VALUE};
    const unique_label = [];
    const _this = this; 
    this.sentences_ = [];

    const formated_objs = objs.data.map(function(value, idx)
    {
      sum.min_x = Math.min(sum.min_x, value[0]);
      sum.max_x = Math.max(sum.max_x, value[0]);
      
      sum.min_y = Math.min(sum.min_y, value[1]);
      sum.max_y = Math.max(sum.max_y, value[1]);

      let label = objs.label[idx];

      if(!unique_label.includes(label))
        unique_label.push(label);

      _this.sentences_.push({sentence_id: objs.ids[idx], label: label});
      return {sentence_id: objs.ids[idx], x: value[0], y: value[1], label: label};
    });

    unique_label.sort();
    LABEL_COLOR_PALETTE = d3.scaleOrdinal(d3.schemeCategory10).domain(unique_label);    
    this.drawer.draw(formated_objs, sum, LABEL_COLOR_PALETTE);
  }
}

class WordView extends VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    super(div_list_id, header_id, chart_id, info_id);
    this.filter_type = "word";
    this.source_type = "token";    
    this.drawer = new WordCloud(chart_id, TOOLTIP);
    this.words = null;
    this.max_samples = 50;
    const _this = this;
    this.drawer.on("end", function(data, position){ _this.drawer_callback(data, position); });
    
    document.getElementById("clear_word_selection").addEventListener("click", function(event){ _this.clear_all(event) }); 
  }
  clear_all(event)
  {
    event.preventDefault();
    this.drawer.clear();
    this.set_header("Token - " + this.max_samples + " samples more frequent");    
  }  
  drawer_callback(data, position)
  {
    const data_filtered = FILTER.set_view(this.filter_type, data).get_view();
    let text_ids = {};

    for(let i = 0; i < data.length; i++)
    {
      if(data_filtered.includes(data[i]) && position[i] !== null)
      {
        if(!Object.keys(text_ids).includes(data[i])) 
          text_ids[data[i]] = [];

        text_ids[data[i]].push( position[i] );
      } 
    }    
    
    PROJECTION_VIEW.select_items(Object.keys(text_ids).length == 0 ? data_filtered : Object.keys(text_ids));
    let text_label = [];

    if(Object.keys(text_ids).length == 0)
      text_label = PROJECTION_VIEW.sentences.filter(function(stn){ return data_filtered.includes(stn.sentence_id); });
    else
      text_label = PROJECTION_VIEW.sentences.filter(function(stn){ return Object.keys(text_ids).includes(stn.sentence_id); });

    EXPLANATION_VIEW.select_items(text_label);
    TEXT_VIEW.select_items(Object.keys(text_ids).length == 0 ? data_filtered : text_ids);
  }
  counter(array)
  {
    let counts = {};

    for(let element of array) 
    {
      counts[element.frequency] = (counts[element.frequency] || 0) + 1;
    }

    return counts;    
  }
  zipf_law(words)
  {
    let filtered = words;
    const min_freq = Math.max(words[0].frequency, 3);
    const max_freq = words[words.length - 1].frequency;
    let counts = this.counter(filtered)
    
    //The filtering process with less than 4 different frequencies doesn't make sense
    if(min_freq !== max_freq && Object.keys(counts).length > 3)
    {
      //Remove lower and upper frequencies
      filtered = words.filter(function(item) { return item.frequency > min_freq && item.frequency < max_freq; });
      const aux = filtered.map(function(item) { return Math.log10(item.frequency); });

      const q1 = d3.quantile(aux, 0.25);
      const q3 = d3.quantile(aux, 0.75);
      const iqr = q3 - q1;

      //Filter out frequencies out [min_freq, max_freq] interval
      filtered = filtered.filter(function(item) { return item.frequency > Math.max(min_freq, 10**(q1 - 0.1 * iqr)) && item.frequency < Math.min(max_freq, 10**(q3 + 1.5 * iqr)); });
    }

    return filtered.length > this.max_samples ? filtered.slice(-this.max_samples) : filtered; 
  }    
  show(data, clean=true)
  {
    const objs = this.extract_data(data.objs);
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
        postag: objs.data.postag[index]   
      };
    })
    .sort(function(a, b) { return a.frequency - b.frequency; })
    
    const filtered = this.zipf_law(this.words);
    this.set_header("Token - " + filtered.length + " samples more frequent");
    
    this.drawer.draw(filtered, {min_freq: filtered[0].frequency, max_freq: filtered[filtered.length - 1].frequency});
  }
  select_items(items, redraw = true)
  {
    let filtered = this.words;

    if (FILTER.count(this.filter_type) != 0)
    {
      filtered = this.words 
        .map(function(item, i) 
        {  
          const new_item = {
            id: item.id, 
            text: item.text,
            frequency: 0, 
            sentences: [],
            position: [], 
            named_entity: [], 
            postag: []     
          };

          item.sentences.forEach(function(stn, j)
          {
            if(items.includes(stn))
            {
              new_item.frequency += 1;
              new_item.sentences.push(stn);
              new_item.position.push(item.position[j]);
              new_item.named_entity.push(item.named_entity[j]);
              new_item.postag.push(item.postag[j]);
            }
          });

          return new_item; 
        })
        .filter(function(value, index, array){ return value.frequency > 0;  })
        .sort(function(a, b) { return a.frequency - b.frequency; });
    }

    filtered = filtered.length === 0 ? filtered : this.zipf_law(filtered);
    this.set_header("Token - " + filtered.length + " samples more frequent");

    if(filtered.length === 0)
      this.drawer.draw(filtered, {min_freq: 0, max_freq: 0});
    else
      this.drawer.draw(filtered, {min_freq: filtered[0].frequency, max_freq: filtered[filtered.length - 1].frequency});
  }    
}

class Explanation extends VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    super(div_list_id, header_id, chart_id, info_id);
    this.filter_type = "explanation";
    this.source_type = "class";
    this.drawer = new SankyDiagram(chart_id, TOOLTIP);
    this.classes = null;
    this.data = null;
    this.name = null;
    this.selected_classes = [];
    this.selected_sentence = [];
    const _this = this;
    this.drawer.on("end", function(data, position){ _this.drawer_callback(data, position); });    
  }
  drawer_callback(data, position)
  {
    let data_filtered = data
      .filter(function(value, index, array) { return array.indexOf(value) === index; });
    data_filtered = FILTER.set_view(this.filter_type, data_filtered).get_view();
    let text_ids = {};

    for(let i = 0; i < data.length; i++)
    {
      if(data_filtered.includes(data[i]))
      {
        if(!Object.keys(text_ids).includes(data[i])) 
          text_ids[data[i]] = [];

        if(position.length > 0)
          text_ids[data[i]].push( position[i] );
      } 
    }  
    
    PROJECTION_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    WORD_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : Object.keys(text_ids));
    TEXT_VIEW.select_items(Object.keys(text_ids) == 0 ? data_filtered : text_ids);
  } 
  process_data(filter_class, filter_sentence)
  {
    const _this = this;
    //List with all nodes (classes + tokens)
    let aux_search = Object.assign([], this.classes);

    if(this.selected_classes.length > 0)
      aux_search = this.classes.filter(function(cls){ return _this.selected_classes.includes(cls); });

    let node_objects = aux_search.map(function(item, i) { return {id: item, type: "class"}; });
    //Avoides classes and tokens with the same name
    aux_search = aux_search.map(function(item, i) { return item + "_class"; });
    //Used to set a fixedValue for classes that don't have connections
    let class_values = [];
    let token = [];
    let links  = [];


    //Create links and tokens for each class row in this.data.explanations.
    for(let i = 0; i < this.data.explanations.length; i++)
    {
      let source_i = aux_search.indexOf(this.classes[i] + "_class");

      if(source_i !== -1)
      {
        //Sorting explanations ids by their values
        let dec_order = this.data.explanations[i]
          .map(function(value, index){ return [index, value];  }) //real index, value
          .sort(function(a, b){ return b[1] - a[1]  }) //compare and order values
          .map(function(value){ return value[0]; });   //return real index

        let max_value = this.data.explanations[i][ dec_order[0] ];
        let min_value = this.data.explanations[i][ dec_order[dec_order.length - 1] ];
        let value = [];

        if(min_value !== max_value)
        {
          let count = 0;

          //Iterating over the class explanations to create links and tokens
          for(let j of dec_order)
          {
            if(count == this.drawer.total_links)
              break;
            
            if(this.data.explanations[i][j] > 0)
            {
              let token_selected_sentences = this.data.sentences[j];

              if(this.selected_sentence.length > 0)
                token_selected_sentences = token_selected_sentences.filter(function(stn, index)
                { 
                  return _this.selected_sentence.includes(stn) && _this.data.label[j][index] === _this.classes[i];  
                });

              if(token_selected_sentences.length > 0)
              {
                count++;

                let target_i = aux_search.indexOf(this.data.tokens[j]);
  
                if (target_i == -1)
                {
                  token.push({
                    id: this.data.tokens[j], 
                    type: "token", 
                    sentences: this.data.sentences[j],  
                    label: this.data.label[j], //ground-truth
                    position: this.data.position[j], 
                    named_entity: this.data.named_entity[j], 
                    postag: this.data.postag[j]
                  });
                  aux_search.push(this.data.tokens[j]);
                  target_i = aux_search.length - 1;
                }  
  
                links.push({source: source_i, target: target_i, value: this.data.explanations[i][j]});
                value.push(this.data.explanations[i][j]);
              }
            }
          }
        }

        class_values.push(value);
      }
    }

    //Sum all the link values of a class. Classes without links are initialized with Number.MAX_VALUE
    let class_min_value = class_values.map(function(value, index)
      {  
        return  value.length == 0 ? Number.MAX_VALUE : value.reduce(function(sum, vl){ return sum + vl; });
      });
    //Mininum class value
    class_min_value = Math.min.apply(null,  class_min_value);
    //Initialize the value of classes without links
    node_objects.forEach(function(cls, index)
    {
      if(class_values[index].length == 0)
        cls.fixedValue = class_min_value / 2;
    });

    this.set_header("Class - " + node_objects.length + " predicted classes - " + this.name);

    return {nodes: node_objects.concat(token), links: links, sentences: PROJECTION_VIEW.sentences};
  }   
  show(data, clean=true)
  { 
    const objs = this.extract_data(data.objs);

    this.classes = objs.ids;
    this.data = objs.data
    this.name = objs.name;

    this.set_header("Class - " + this.classes.length + " predicted classes - " + this.name);
    
    this.drawer.draw(this.process_data(), {}, LABEL_COLOR_PALETTE);
  }
  select_items(items, redraw = true)    
  {
    const _this = this;
    this.selected_classes = [];
    this.selected_sentence = [];

    if(FILTER.count(this.filter_type) != 0)
    {
      items.forEach(function(obj)
      {
        if(!_this.selected_sentence.includes(obj.sentence_id))
          _this.selected_sentence.push(obj.sentence_id);
        if(!_this.selected_classes.includes(obj.label))
          _this.selected_classes.push(obj.label);
      });
    }
    
    this.drawer.draw(this.process_data(), {}, LABEL_COLOR_PALETTE);
  }  
  get_info()
  {
    if(this.data !== null)
    {
      let html = "";

      html += "<div class='table-responsive'>";
      html += "<table class='table table-bordered' id='dataTable' width='100%' cellspacing='0'>";
      html += "<caption class='table-caption'>Classification report</caption>";
      html += "<thead><tr><th>class</th><th>precision</th><th>recall</th><th>f1</th></tr></thead>";
      html += "<tbody>";

      let precision = 0;
      let recall = 0;
      let f1_score = 0;

      for(let i = 0; i < this.data.class_report.length; i++)
      {
        html += "<tr>";

        html += "<td>" + this.classes[i] + "</td>";
        html += "<td>" + this.data.class_report[i][0].toFixed(2) + "</td>"; //precision
        html += "<td>" + this.data.class_report[i][1].toFixed(2) + "</td>"; //recall
        html += "<td>" + this.data.class_report[i][2].toFixed(2) + "</td>"; //f1-score

        html += "</tr>";

        precision += this.data.class_report[i][0];
        recall    += this.data.class_report[i][1];
        f1_score  += this.data.class_report[i][2];
      }

      precision /= this.data.class_report.length; 
      recall    /= this.data.class_report.length;
      f1_score  /= this.data.class_report.length;

      html += "</tbody>";

      html += "<tfoot>";

      html += "<tr>";
      html += "<th>avg.</th>";
      html += "<td>" + precision.toFixed(2) + "</td>"; //precision
      html += "<td>" + recall.toFixed(2)    + "</td>"; //recall
      html += "<td>" + f1_score.toFixed(2)  + "</td>"; //f1-score
      html += "</tr>";

      html += "<tr>";
      html += "<th colspan='3'>acc.</th>";
      html += "<td>" + this.data.class_report[0][3].toFixed(2) + "</td>";
      html += "</tr>";      

      html += "</tfoot>";
      html += "</table>"
      html += "</div>";

      let infidelity = this.data.exp_report.reduce((accumulator, value) => accumulator + value, 0) / this.data.exp_report.length;
      let std = this.data.exp_report.reduce((accumulator, value) => accumulator + (value - infidelity) ** 2, 0);
      std = Math.sqrt( std / (this.data.exp_report.length - 1) );

      html += "<div class='table-responsive'>";
      html += "<table class='table table-bordered' id='dataTable' width='100%' cellspacing='0'>";
      html += "<caption class='table-caption'>Explanation report</caption>";
      html += "<thead><tr><th></th><th>infidelity</th></tr></thead>";
      html += "<tbody>";
      html += "<th>avg&plusmn;std</th>";
      html += "<td>" + infidelity.toFixed(4) + "&plusmn;" + std.toFixed(4) + "</td>";
      html += "</tbody>";      

      return html;
    }
    
    return null;
  }
}

class TextView extends VisManager
{
  constructor(div_list_id, header_id, chart_id, info_id)
  {
    super(div_list_id, header_id, chart_id, info_id);
    this.filter_type = "text";
    this.objs = null;
    this.highlight = new HighlightText();    
    this.paginator = new PaginateText("text-previous", "text-next", "text-first", "text-last", "text-position", chart_id); 

    const _this = this;

    this.paginator.on("paginate", function(obj, text_tokens)
    {
      _this.highlight.execute(obj.dataset.text, _this.get_unique_token(text_tokens), obj.querySelectorAll("span")[1]);
    });

    document.getElementById("clear_text_selection").addEventListener("click", function(event){ _this.clear_all(event) }); 
  }
  show(data, clean=true)
  { 
    const _this = this;
    this.objs  = this.extract_data(data.objs);    
    this.label = this.objs.data.label;
    const row = document.createElement("div");
    row.className = "row overflow-auto";

    this.div_chart.innerHTML = "";
    this.div_chart.appendChild(row);

    this.paginator.count_text = this.objs.data.text.length;

    for (let idx = 0; idx < this.objs.data.text.length; idx++)
    {
      let col = document.createElement("div");
      col.className = "col-xl-12 col-md-6 mb-4"
      col.dataset.id = this.objs.ids[idx];
      col.dataset.text = this.objs.data.text[idx];

      let card = document.createElement("div");
      card.className = "card border-left-primary h-100 py-2";
      card.style.cssText = "border-left-color: " + LABEL_COLOR_PALETTE(this.objs.data.label[idx]) + " !important";

      let body = document.createElement("div");
      body.className = "card-body";

      let title = document.createElement("span");
      title.className = "font-weight-bold text-id";
      title.innerHTML = this.objs.ids[idx] + " (" + this.objs.data.label[idx] + "):";
      title.data = {id: this.objs.ids[idx], label: this.objs.data.label[idx]};

      let text = document.createElement("span");
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
    this.set_header("Text - " + this.paginator.get_count_text() + " samples");
  }
  clear_all(event)
  {
    event.preventDefault();
    const _this = this;
    document.querySelectorAll(".text-selected").forEach(function(element)
    {
      element.classList.remove("text-selected");   
      let text_id = FILTER.set_view(_this.filter_type, []).get_view();

      PROJECTION_VIEW.select_items(text_id);
      WORD_VIEW.select_items(text_id);
      text_id = PROJECTION_VIEW.sentences.filter(function(stn){ return text_id.includes(stn.sentence_id); });
      EXPLANATION_VIEW.select_items(text_id);    
    });
    
    this.set_header("Text - " + this.paginator.get_count_text() + " samples");
  }
  drawer_callback(event)
  {
    event.preventDefault();

    let text_id = FILTER.get_view(this.filter_type);

    if(event.target.classList.contains("text-selected"))
    {
      event.target.classList.remove("text-selected");   
      let index = text_id.indexOf(event.target.data.id);
      text_id.splice(index, 1);
    }  
    else 
    {
      event.target.classList.add("text-selected");
      text_id.push(event.target.data.id);
    }  

    text_id = FILTER.set_view(this.filter_type, text_id).get_view();

    PROJECTION_VIEW.select_items(text_id);
    WORD_VIEW.select_items(text_id);
    text_id = PROJECTION_VIEW.sentences.filter(function(stn){ return text_id.includes(stn.sentence_id); });
    EXPLANATION_VIEW.select_items(text_id);    
  }
  get_unique_token(text_tokens)
  {
    let aux = [];

    for(let token of text_tokens)
    {
      let found_index = -1;

      for(let kdx = 0; kdx < aux.length; kdx++)
      {
        if(aux[kdx][0] === token[0] && aux[kdx][1][0] === token[1][0] && aux[kdx][1][1] === token[1][1])
        {
          found_index = kdx;
          break;
        }
      }

      if(found_index === -1)
        aux.push(token);
    }

    return aux;
  }
  select_items(items)
  {
    this.paginator.set_selected_text(FILTER.count(this.filter_type) === 0 ? [] : items);
    this.paginator.paginate_text();
    this.paginator.manage_control();
    this.set_header("Text - " + this.paginator.get_count_text() + " samples");
  }      
}

class HighlightText
{
  execute(original_text, text_tokens, html_tag)
  { 
    //text_tokens: list of [token, [start pos, end pos]]
    text_tokens.sort(function(a, b) { return b[1][0] - a[1][0]; });
    const unique_token_id = text_tokens
      .map(function(obj) { return obj === null ? obj.trim().toLowerCase() : obj[0].trim().toLowerCase(); })
      .filter(function(value, index, array){ return array.indexOf(value) === index; })
      .sort();    
    const palette = d3.scaleOrdinal(d3.schemeSet3).domain(unique_token_id);
    let new_text = original_text;

    for(let tkn_pos of text_tokens)  
    {
      let before = new_text.slice(0, tkn_pos[1][0]);
      let after  = new_text.slice(tkn_pos[1][1]);
      let token  = new_text.substring(tkn_pos[1][0], tkn_pos[1][1]);

      new_text = before + "<span style='background-color:" + palette(tkn_pos[0].trim().toLowerCase()) + "'>" + token + "</span>" + after;
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
    if (Object.keys(this.call_back).includes(type))
      this.call_back[type] = call_back;

    return this;
  }  
  get_count_text()
  {
    return this.count_selected_text > 0 ? this.count_selected_text : this.count_text;
  }
  bind_page_click_event(element, call_back)
  {
    const _this = this;

    element.addEventListener("click", function(event)
    {
      event.preventDefault();
      call_back();

      _this.manage_control();
      _this.paginate_text();
    });     
  }   
  set_event()
  {
    const _this = this;

    this.bind_page_click_event(this.first_page, function(){  _this.first_idx = 0; });
    this.bind_page_click_event(this.previous_page, function(){  _this.first_idx -= _this.inc; });
    this.bind_page_click_event(this.next_page, function(){  _this.first_idx += _this.inc; });
    this.bind_page_click_event(this.last_page, function()
    { 
      const aux_idx = Math.floor(_this.get_count_text() / _this.inc) * _this.inc;
      _this.first_idx = aux_idx == _this.get_count_text() ? _this.get_count_text() - _this.inc : aux_idx;
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

    for(let idx = 0; idx < cols.length; idx++)
    {
      this.call_back.paginate(cols[idx], []);
      let hidden = false;

      //Selection from: scatterplot and heatmap
      if(this.selected_text instanceof Array)
      {
        hidden = this.selected_text.length > 0 && !this.selected_text.includes(cols[idx].dataset.id);
      }
      //Selection from: treemap
      else if(ids.length > 0)
      {
        hidden = !ids.includes(cols[idx].dataset.id);

        if(!hidden)
        {
          let text_tokens = this.selected_text[cols[idx].dataset.id];
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

    this.toggle_controls([this.first_page, this.previous_page], this.first_idx === 0);
    this.toggle_controls([this.next_page, this.last_page], this.first_idx + this.inc >= this.get_count_text());
  }
  toggle_controls(controls, disabled) 
  {
    controls.forEach(function(el)
    {
      el.classList.toggle("disabled", disabled);
    });
  }         
}
